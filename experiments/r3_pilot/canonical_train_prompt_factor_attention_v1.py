"""Target-free prompt-conditioned attention over static factor carrier scans.

The caller supplies only the physical prompt observations accepted by the
static prompt V4 capsule.  This module authenticates the prompt winner trace,
deduplicates its physical span assemblies, enumerates every generic selector
geometry, and runs the static factor V4 capsule internally.  Twenty-four
independent simulated cells then require coincidence between learned prompt
production/semantic-slot support and scanned factor membership.

No caller-supplied factor observation, desired output, program, task, family,
target, or expected factor enters the mechanism.  Generic ``ascii-class``
prompt support activates the complete physical class bank; the host never picks
a concrete class label.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from typing import NamedTuple

import numpy as np
from experiments.r3_pilot import (
    public_train_factor_workspace_v4_static_checkpoint_capsule_v1 as factor_checkpoint,
)
from experiments.r3_pilot import (
    public_train_prompt_selection_v4_static_checkpoint_capsule_v1 as prompt_checkpoint,
)

SCHEMA = "canonical-train-prompt-factor-attention-v1"
ATTENTION_READY = "CANONICAL_TRAIN_PROMPT_FACTOR_ATTENTION_V1_READY"
SCAN_READY = "CANONICAL_TRAIN_PROMPT_FACTOR_CARRIER_SCAN_V1_READY"
STOP_PROMPT = "STOP_CANONICAL_TRAIN_PROMPT_FACTOR_ATTENTION_V1_PROMPT"
STOP_ZERO = "STOP_CANONICAL_TRAIN_PROMPT_FACTOR_ATTENTION_V1_ZERO"
STOP_CAP = "STOP_CANONICAL_TRAIN_PROMPT_FACTOR_ATTENTION_V1_CAP"
STOP_COLLISION = "STOP_CANONICAL_TRAIN_PROMPT_FACTOR_ATTENTION_V1_COLLISION"
STOP_CONNECTOR = "STOP_CANONICAL_TRAIN_PROMPT_FACTOR_ATTENTION_V1_PROMPT_CELL_CONNECTOR"
STOP_DERANGEMENT = "STOP_CANONICAL_TRAIN_PROMPT_FACTOR_ATTENTION_V1_DERANGEMENT"
STOP_CELL_OFF = "STOP_CANONICAL_TRAIN_PROMPT_FACTOR_ATTENTION_V1_CELL_OFF"

FACTOR_CAPACITY = 4
SELECTOR_GEOMETRY_COUNT = 8
PROMPT_CURRENT_PA = 1_200.0
FACTOR_CURRENT_PA = 1_200.0
COINCIDENCE_THRESHOLD_PA = 2_000.0

_NAMESPACE = b"canonical-train-prompt-factor-attention-v1\0"
_FACTOR_INDEX = {name: index for index, name in enumerate(factor_checkpoint.FACTOR_REGISTRY)}
_CLASS_FACTOR_INDICES = tuple(
    index
    for index, name in enumerate(factor_checkpoint.FACTOR_REGISTRY)
    if name.startswith("class:")
)


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        _plain(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _digest(label: str, value: object) -> str:
    result = sha256()
    result.update(_NAMESPACE)
    result.update(label.encode("utf-8"))
    result.update(b"\0")
    result.update(_canonical_json(value))
    return result.hexdigest()


class _DiscreteNeuronPopulation:
    """Reset-on-spike current-integrating neurons for independent coincidence."""

    __slots__ = ("_membrane", "_threshold_pa")

    def __init__(self, size: int, *, threshold_pa: float) -> None:
        if type(size) is not int or size < 1 or threshold_pa <= 0.0:
            raise ValueError("attention population requires a positive domain")
        self._membrane = np.zeros(size, dtype=np.float64)
        self._threshold_pa = float(threshold_pa)

    def step(self, currents_pa: np.ndarray) -> np.ndarray:
        currents = np.asarray(currents_pa, dtype=np.float64)
        if currents.shape != self._membrane.shape or not np.all(np.isfinite(currents)):
            raise ValueError("attention current domain changed")
        self._membrane += currents
        spikes = self._membrane >= self._threshold_pa
        self._membrane[spikes] = 0.0
        return np.asarray(spikes, dtype=np.bool_)


class _Intervention(NamedTuple):
    name: str
    factor_rotation: int
    cell_gain: float
    collide_cell_ids: bool


_TREATMENT = _Intervention("treatment", 0, 1.0, False)
_DERANGED = _Intervention("deranged-factor-membership", 12, 1.0, False)
_CELL_OFF = _Intervention("cell-off", 0, 0.0, False)
_COLLISION = _Intervention("coincidence-cell-collision", 0, 1.0, True)


@dataclass(frozen=True, slots=True)
class FactorCarrierScanAudit:
    status: str
    raw_span_count: int
    unique_assembly_count: int
    selector_geometry_count: int
    factor_inference_attempts: int
    ready_observation_count: int
    zero_observation_count: int
    cap_observation_count: int
    collision_observation_count: int
    factor_event_counts: tuple[int, ...]
    physical_membership: tuple[bool, ...]
    duplicate_equivalent_collapses: int
    certificate_root_sha256: str
    sha256: str


@dataclass(frozen=True, slots=True)
class PromptFactorAttentionState:
    status: str
    stop_reason: str | None
    intervention: str
    prompt_checkpoint_sha256: str
    prompt_artifact_sha256: str
    prompt_input_sha256: str
    prompt_actual_winner_sha256: str
    prompt_certificate_sha256: str
    prompt_output_registry_sha256: str
    prompt_factor_connector_sha256: str
    prompt_factor_connector_cell_count: int
    runtime_winner_relay_parse_count: int
    factor_checkpoint_sha256: str
    factor_artifact_sha256: str
    prompt_relay_count: int
    prompt_support_event_counts: tuple[int, ...]
    prompt_physical_support: tuple[bool, ...]
    prompt_duplicate_equivalent_collapses: int
    generic_ascii_class_support: bool
    host_class_label_selection_count: int
    scan: FactorCarrierScanAudit
    projected_factor_membership: tuple[bool, ...]
    coincidence_cell_ids: tuple[int, ...]
    coincidence_currents_pa_hex: tuple[str, ...]
    physical_coincidence_spikes: tuple[bool, ...]
    factor_membership: tuple[str, ...]
    membership_count: int
    caller_factor_observation_count: int
    target_input_count: int
    expected_input_count: int
    desired_input_count: int
    program_input_count: int
    task_input_count: int
    family_input_count: int
    heldout_input_count: int
    protected_input_count: int
    executable_learner_import_count: int
    target_protocol_import_count: int
    sha256: str


class _AttentionReceipt(NamedTuple):
    authority: object
    state: PromptFactorAttentionState
    sha256: str


def _scan_payload(scan: FactorCarrierScanAudit) -> dict[str, object]:
    return {
        field: getattr(scan, field)
        for field in scan.__dataclass_fields__
        if field != "sha256"
    }


def _state_payload(state: PromptFactorAttentionState) -> dict[str, object]:
    return {
        "status": state.status,
        "stop_reason": state.stop_reason,
        "intervention": state.intervention,
        "prompt_checkpoint_sha256": state.prompt_checkpoint_sha256,
        "prompt_artifact_sha256": state.prompt_artifact_sha256,
        "prompt_input_sha256": state.prompt_input_sha256,
        "prompt_actual_winner_sha256": state.prompt_actual_winner_sha256,
        "prompt_certificate_sha256": state.prompt_certificate_sha256,
        "prompt_output_registry_sha256": state.prompt_output_registry_sha256,
        "prompt_factor_connector_sha256": state.prompt_factor_connector_sha256,
        "prompt_factor_connector_cell_count": (
            state.prompt_factor_connector_cell_count
        ),
        "runtime_winner_relay_parse_count": state.runtime_winner_relay_parse_count,
        "factor_checkpoint_sha256": state.factor_checkpoint_sha256,
        "factor_artifact_sha256": state.factor_artifact_sha256,
        "prompt_relay_count": state.prompt_relay_count,
        "prompt_support_event_counts": state.prompt_support_event_counts,
        "prompt_physical_support": state.prompt_physical_support,
        "prompt_duplicate_equivalent_collapses": (
            state.prompt_duplicate_equivalent_collapses
        ),
        "generic_ascii_class_support": state.generic_ascii_class_support,
        "host_class_label_selection_count": state.host_class_label_selection_count,
        "scan_sha256": state.scan.sha256,
        "projected_factor_membership": state.projected_factor_membership,
        "coincidence_cell_ids": state.coincidence_cell_ids,
        "coincidence_currents_pa_hex": state.coincidence_currents_pa_hex,
        "physical_coincidence_spikes": state.physical_coincidence_spikes,
        "factor_membership": state.factor_membership,
        "membership_count": state.membership_count,
        "caller_factor_observation_count": state.caller_factor_observation_count,
        "target_input_count": state.target_input_count,
        "expected_input_count": state.expected_input_count,
        "desired_input_count": state.desired_input_count,
        "program_input_count": state.program_input_count,
        "task_input_count": state.task_input_count,
        "family_input_count": state.family_input_count,
        "heldout_input_count": state.heldout_input_count,
        "protected_input_count": state.protected_input_count,
        "executable_learner_import_count": state.executable_learner_import_count,
        "target_protocol_import_count": state.target_protocol_import_count,
    }


def _make_receipt_authority() -> tuple[
    Callable[[PromptFactorAttentionState], _AttentionReceipt],
    Callable[[_AttentionReceipt], bool],
]:
    authority = object()
    issued: dict[int, _AttentionReceipt] = {}

    def issue(state: PromptFactorAttentionState) -> _AttentionReceipt:
        partial = _AttentionReceipt(authority, state, "")
        receipt = partial._replace(
            sha256=_digest(
                "attention-authority-receipt",
                {"state_sha256": state.sha256, "schema": SCHEMA},
            )
        )
        issued[id(receipt)] = receipt
        return receipt

    def authenticate(receipt: _AttentionReceipt) -> bool:
        return bool(
            type(receipt) is _AttentionReceipt
            and issued.get(id(receipt)) is receipt
            and receipt.authority is authority
            and receipt.state.scan.sha256
            == _digest("factor-carrier-scan", _scan_payload(receipt.state.scan))
            and receipt.state.sha256
            == _digest("prompt-factor-attention", _state_payload(receipt.state))
            and receipt.sha256
            == _digest(
                "attention-authority-receipt",
                {"state_sha256": receipt.state.sha256, "schema": SCHEMA},
            )
        )

    return issue, authenticate


_issue_attention_receipt, _authenticate_attention_receipt = _make_receipt_authority()


class PromptFactorAttentionHandoff:
    """Opaque authority issued only by executed prompt and factor inference."""

    __slots__ = ("_receipt",)

    def __init__(self, receipt: _AttentionReceipt) -> None:
        if not _authenticate_attention_receipt(receipt):
            raise TypeError("attention handoff requires a private executed receipt")
        self._receipt = receipt

    @property
    def state(self) -> PromptFactorAttentionState:
        return self._receipt.state

    @property
    def status(self) -> str:
        return self._receipt.state.status

    @property
    def sha256(self) -> str:
        return self._receipt.sha256


def _normalize_spans(
    rows: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    required = {
        "start_token",
        "stop_token",
        "first_spike_step",
        "assembly_cells",
        "spike_cells",
    }
    normalized: list[dict[str, object]] = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != required:
            raise ValueError("physical span record schema changed")
        normalized.append(
            {
                "start_token": int(row["start_token"]),
                "stop_token": int(row["stop_token"]),
                "first_spike_step": int(row["first_spike_step"]),
                "assembly_cells": tuple(int(cell) for cell in row["assembly_cells"]),
                "spike_cells": tuple(int(cell) for cell in row["spike_cells"]),
            }
        )
    return tuple(normalized)


def _normalize_literals(
    rows: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    required = {"canonical_token_index", "typed_output_active_cells"}
    normalized: list[dict[str, object]] = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != required:
            raise ValueError("typed literal record schema changed")
        normalized.append(
            {
                "canonical_token_index": int(row["canonical_token_index"]),
                "typed_output_active_cells": tuple(
                    int(cell) for cell in row["typed_output_active_cells"]
                ),
            }
        )
    return tuple(normalized)


def _prompt_stop_reason(
    certificate: prompt_checkpoint.PromptStructuralCertificate,
) -> str | None:
    if certificate.checkpoint_sha256 != prompt_checkpoint.BANKED_CHECKPOINT_SHA256:
        raise RuntimeError("attention received a different prompt checkpoint")
    terminal = certificate.terminal_winner
    if certificate.status != prompt_checkpoint.STOP:
        return f"physical prompt inference terminated as {certificate.status}"
    if terminal is None or terminal.decision != prompt_checkpoint.STOP:
        return "physical prompt inference lacks its terminal STOP winner"
    if not certificate.occurrences:
        return "physical prompt inference emitted no structural occurrence"
    return None


class _PromptFactorConnector(NamedTuple):
    authority: object
    checkpoint_sha256: str
    artifact_sha256: str
    content_root_sha256: str
    output_registry_sha256: str
    relay_targets: tuple[str, ...]
    factor_indices_by_cell: tuple[tuple[int, ...], ...]
    sha256: str


def _connector_target_factor_indices(target_id: str) -> tuple[int, ...]:
    """Compile one authenticated output-cell target into fixed synapses."""

    parts = target_id.split("|")
    direct: set[int] = set()
    generic_ascii_class = False
    for index, part in enumerate(parts[:-1]):
        value = parts[index + 1]
        if part == "production" and value in _FACTOR_INDEX:
            factor_index = _FACTOR_INDEX[value]
            if factor_index in _CLASS_FACTOR_INDICES:
                generic_ascii_class = True
            else:
                direct.add(factor_index)
        if part == "semantic-slot" and value == "ascii-class":
            generic_ascii_class = True
    if generic_ascii_class:
        direct.update(_CLASS_FACTOR_INDICES)
    return tuple(sorted(direct))


def _connector_payload(value: _PromptFactorConnector) -> dict[str, object]:
    return {
        "checkpoint_sha256": value.checkpoint_sha256,
        "artifact_sha256": value.artifact_sha256,
        "content_root_sha256": value.content_root_sha256,
        "output_registry_sha256": value.output_registry_sha256,
        "connections": tuple(
            (cell_id, target, indices)
            for cell_id, (target, indices) in enumerate(
                zip(
                    value.relay_targets,
                    value.factor_indices_by_cell,
                    strict=True,
                )
            )
        ),
    }


def _make_prompt_factor_connector_authority() -> tuple[
    Callable[[], _PromptFactorConnector],
    Callable[[_PromptFactorConnector], bool],
]:
    authority = object()
    issued: dict[int, _PromptFactorConnector] = {}

    @lru_cache(maxsize=1)
    def issue() -> _PromptFactorConnector:
        capsule = prompt_checkpoint.load_checkpoint_capsule()
        cells = capsule._output_cells
        if tuple(cell.cell_id for cell in cells) != tuple(range(len(cells))):
            raise RuntimeError("prompt output-cell registry identity changed")
        relay_targets = tuple(cell.target_id for cell in cells)
        factor_indices = tuple(
            _connector_target_factor_indices(target) for target in relay_targets
        )
        partial = _PromptFactorConnector(
            authority=authority,
            checkpoint_sha256=capsule.checkpoint_sha256,
            artifact_sha256=capsule.artifact_sha256,
            content_root_sha256=capsule.content_root,
            output_registry_sha256=str(
                capsule._metadata["output_registry_sha256"]
            ),
            relay_targets=relay_targets,
            factor_indices_by_cell=factor_indices,
            sha256="",
        )
        connector = partial._replace(
            sha256=_digest("prompt-cell-factor-connectors", _connector_payload(partial))
        )
        issued[id(connector)] = connector
        return connector

    def authenticate(value: _PromptFactorConnector) -> bool:
        return bool(
            type(value) is _PromptFactorConnector
            and issued.get(id(value)) is value
            and value.authority is authority
            and value.checkpoint_sha256
            == prompt_checkpoint.BANKED_CHECKPOINT_SHA256
            and value.artifact_sha256
            == prompt_checkpoint.EXPECTED_ARTIFACT_SHA256
            and value.content_root_sha256
            == prompt_checkpoint.EXPECTED_ARTIFACT_CONTENT_ROOT
            and len(value.relay_targets) == len(value.factor_indices_by_cell)
            and value.sha256
            == _digest("prompt-cell-factor-connectors", _connector_payload(value))
        )

    return issue, authenticate


_prompt_factor_connector, _authenticate_prompt_factor_connector = (
    _make_prompt_factor_connector_authority()
)


def _prompt_support(
    certificate: prompt_checkpoint.PromptStructuralCertificate,
    connector: _PromptFactorConnector,
) -> tuple[tuple[int, ...], tuple[bool, ...], int, bool, int, int]:
    if not _authenticate_prompt_factor_connector(connector):
        raise RuntimeError("prompt factor connector authentication failed")
    event_counts = [0] * factor_checkpoint.FACTOR_COUNT
    relay_count = 0
    collision_count = 0
    connector_mismatch_count = 0
    generic_ascii_class = False
    for occurrence in certificate.occurrences:
        for winner in occurrence.branch_winners:
            if winner.decision != "RELAY":
                continue
            relay_count += 1
            cell_id = winner.selected_cell_id
            if (
                type(cell_id) is not int
                or not 0 <= cell_id < len(connector.relay_targets)
                or winner.relay_id != connector.relay_targets[cell_id]
            ):
                connector_mismatch_count += 1
                continue
            indices = connector.factor_indices_by_cell[cell_id]
            class_indices = tuple(
                index for index in indices if index in _CLASS_FACTOR_INDICES
            )
            ordinary_indices = tuple(
                index for index in indices if index not in _CLASS_FACTOR_INDICES
            )
            generic_ascii_class = generic_ascii_class or bool(class_indices)
            if len(ordinary_indices) > 1:
                collision_count += 1
            for factor_index in indices:
                event_counts[factor_index] += 1
    physical_support = tuple(count > 0 for count in event_counts)
    return (
        tuple(event_counts),
        physical_support,
        relay_count,
        generic_ascii_class,
        collision_count,
        connector_mismatch_count,
    )


def _unique_authenticated_assemblies(
    spans: tuple[dict[str, object], ...],
) -> tuple[tuple[int, ...], ...] | None:
    assemblies: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for span in spans:
        assembly = tuple(int(cell) for cell in span["assembly_cells"])
        spikes = tuple(int(cell) for cell in span["spike_cells"])
        if (
            len(assembly) != factor_checkpoint.ASSEMBLY_ACTIVE_CELL_COUNT
            or tuple(sorted(assembly)) != assembly
            or len(set(assembly)) != len(assembly)
            or any(not 0 <= cell < factor_checkpoint.LEXICAL_CELL_COUNT for cell in assembly)
            or spikes != assembly
        ):
            return None
        if assembly not in seen:
            seen.add(assembly)
            assemblies.append(assembly)
    return tuple(assemblies)


def _make_scan(
    *,
    status: str,
    raw_span_count: int,
    unique_assembly_count: int,
    attempts: int,
    ready: int,
    zero: int,
    cap: int,
    collision: int,
    event_counts: tuple[int, ...],
    certificate_sha256s: tuple[str, ...],
) -> FactorCarrierScanAudit:
    membership = tuple(count > 0 for count in event_counts)
    partial = FactorCarrierScanAudit(
        status=status,
        raw_span_count=raw_span_count,
        unique_assembly_count=unique_assembly_count,
        selector_geometry_count=SELECTOR_GEOMETRY_COUNT,
        factor_inference_attempts=attempts,
        ready_observation_count=ready,
        zero_observation_count=zero,
        cap_observation_count=cap,
        collision_observation_count=collision,
        factor_event_counts=event_counts,
        physical_membership=membership,
        duplicate_equivalent_collapses=sum(event_counts) - sum(membership),
        certificate_root_sha256=_digest(
            "factor-scan-certificates", certificate_sha256s
        ),
        sha256="",
    )
    return FactorCarrierScanAudit(
        **_scan_payload(partial),
        sha256=_digest("factor-carrier-scan", _scan_payload(partial)),
    )


def _scan_factor_carriers(
    spans: tuple[dict[str, object], ...],
) -> FactorCarrierScanAudit:
    assemblies = _unique_authenticated_assemblies(spans)
    if assemblies is None:
        return _make_scan(
            status=STOP_COLLISION,
            raw_span_count=len(spans),
            unique_assembly_count=0,
            attempts=0,
            ready=0,
            zero=0,
            cap=0,
            collision=1,
            event_counts=(0,) * factor_checkpoint.FACTOR_COUNT,
            certificate_sha256s=(),
        )
    event_counts = [0] * factor_checkpoint.FACTOR_COUNT
    certificate_sha256s: list[str] = []
    ready = zero = cap = collision = attempts = 0
    for assembly in assemblies:
        for direction in ("NEXT", "PREV"):
            for cue_slot in range(4):
                selected_slot = (
                    (cue_slot + 1) % 4
                    if direction == "NEXT"
                    else (cue_slot - 1) % 4
                )
                observation = factor_checkpoint.RawSpanObservation(
                    direction=direction,
                    cue_slot=cue_slot,
                    selected_slot=selected_slot,
                    assembly_cells=assembly,
                )
                certificate = factor_checkpoint.infer_factors((observation,))
                attempts += 1
                certificate_sha256s.append(certificate.sha256)
                if (
                    certificate.checkpoint_sha256
                    != factor_checkpoint.BANKED_CHECKPOINT_SHA256
                    or certificate.artifact_sha256 != factor_checkpoint.ARTIFACT_SHA256
                ):
                    raise RuntimeError("attention factor checkpoint identity changed")
                if certificate.status == "READY_FACTOR_SET":
                    ready += 1
                    for index, occupied in enumerate(certificate.physical_occupancy):
                        event_counts[index] += int(occupied)
                elif certificate.status == "STOP_ZERO":
                    zero += 1
                elif certificate.status == "STOP_EVENT_CAPACITY_4":
                    cap += 1
                else:
                    collision += 1
    status = STOP_CAP if cap else STOP_COLLISION if collision else SCAN_READY
    return _make_scan(
        status=status,
        raw_span_count=len(spans),
        unique_assembly_count=len(assemblies),
        attempts=attempts,
        ready=ready,
        zero=zero,
        cap=cap,
        collision=collision,
        event_counts=tuple(event_counts),
        certificate_sha256s=tuple(certificate_sha256s),
    )


def _empty_scan(raw_span_count: int) -> FactorCarrierScanAudit:
    return _make_scan(
        status=STOP_PROMPT,
        raw_span_count=raw_span_count,
        unique_assembly_count=0,
        attempts=0,
        ready=0,
        zero=0,
        cap=0,
        collision=0,
        event_counts=(0,) * factor_checkpoint.FACTOR_COUNT,
        certificate_sha256s=(),
    )


def _build_state(
    *,
    status: str,
    stop_reason: str | None,
    intervention: _Intervention,
    prompt: prompt_checkpoint.PromptStructuralCertificate,
    connector: _PromptFactorConnector,
    prompt_event_counts: tuple[int, ...],
    prompt_support: tuple[bool, ...],
    prompt_relay_count: int,
    generic_ascii_class: bool,
    scan: FactorCarrierScanAudit,
    projected_membership: tuple[bool, ...],
    cell_ids: tuple[int, ...],
    currents: tuple[float, ...],
    spikes: tuple[bool, ...],
) -> PromptFactorAttentionState:
    factor_membership = (
        tuple(
            factor
            for factor, spiked in zip(
                factor_checkpoint.FACTOR_REGISTRY, spikes, strict=True
            )
            if spiked
        )
        if status == ATTENTION_READY
        else ()
    )
    partial = PromptFactorAttentionState(
        status=status,
        stop_reason=stop_reason,
        intervention=intervention.name,
        prompt_checkpoint_sha256=prompt.checkpoint_sha256,
        prompt_artifact_sha256=prompt_checkpoint.EXPECTED_ARTIFACT_SHA256,
        prompt_input_sha256=prompt.input_sha256,
        prompt_actual_winner_sha256=prompt.actual_winner_sha256,
        prompt_certificate_sha256=prompt.sha256,
        prompt_output_registry_sha256=connector.output_registry_sha256,
        prompt_factor_connector_sha256=connector.sha256,
        prompt_factor_connector_cell_count=len(connector.relay_targets),
        runtime_winner_relay_parse_count=0,
        factor_checkpoint_sha256=factor_checkpoint.BANKED_CHECKPOINT_SHA256,
        factor_artifact_sha256=factor_checkpoint.ARTIFACT_SHA256,
        prompt_relay_count=prompt_relay_count,
        prompt_support_event_counts=prompt_event_counts,
        prompt_physical_support=prompt_support,
        prompt_duplicate_equivalent_collapses=(
            sum(prompt_event_counts) - sum(prompt_support)
        ),
        generic_ascii_class_support=generic_ascii_class,
        host_class_label_selection_count=0,
        scan=scan,
        projected_factor_membership=projected_membership,
        coincidence_cell_ids=cell_ids,
        coincidence_currents_pa_hex=tuple(value.hex() for value in currents),
        physical_coincidence_spikes=spikes,
        factor_membership=factor_membership,
        membership_count=(sum(spikes) if status == ATTENTION_READY else 0),
        caller_factor_observation_count=0,
        target_input_count=0,
        expected_input_count=0,
        desired_input_count=0,
        program_input_count=0,
        task_input_count=0,
        family_input_count=0,
        heldout_input_count=0,
        protected_input_count=0,
        executable_learner_import_count=0,
        target_protocol_import_count=0,
        sha256="",
    )
    return PromptFactorAttentionState(
        **{
            field: getattr(partial, field)
            for field in partial.__dataclass_fields__
            if field != "sha256"
        },
        sha256=_digest("prompt-factor-attention", _state_payload(partial)),
    )


def _run_attention(
    span_records: Sequence[Mapping[str, object]],
    literal_records: Sequence[Mapping[str, object]],
    signature_role_record: Mapping[str, object],
    intervention: _Intervention,
) -> PromptFactorAttentionHandoff:
    spans = _normalize_spans(span_records)
    literals = _normalize_literals(literal_records)
    signature_role = _plain(signature_role_record)
    prompt = prompt_checkpoint.infer_prompt(spans, literals, signature_role)
    connector = _prompt_factor_connector()
    prompt_reason = _prompt_stop_reason(prompt)
    empty_bool = (False,) * factor_checkpoint.FACTOR_COUNT
    empty_int = (0,) * factor_checkpoint.FACTOR_COUNT
    cell_ids = tuple(range(factor_checkpoint.FACTOR_COUNT))
    if prompt_reason is not None:
        state = _build_state(
            status=STOP_PROMPT,
            stop_reason=prompt_reason,
            intervention=intervention,
            prompt=prompt,
            connector=connector,
            prompt_event_counts=empty_int,
            prompt_support=empty_bool,
            prompt_relay_count=0,
            generic_ascii_class=False,
            scan=_empty_scan(len(spans)),
            projected_membership=empty_bool,
            cell_ids=cell_ids,
            currents=(0.0,) * factor_checkpoint.FACTOR_COUNT,
            spikes=empty_bool,
        )
        return PromptFactorAttentionHandoff(_issue_attention_receipt(state))

    (
        prompt_events,
        prompt_support,
        relay_count,
        ascii_class,
        prompt_collisions,
        connector_mismatches,
    ) = _prompt_support(prompt, connector)
    if connector_mismatches:
        state = _build_state(
            status=STOP_CONNECTOR,
            stop_reason="prompt relay text disagreed with its selected output-cell identity",
            intervention=intervention,
            prompt=prompt,
            connector=connector,
            prompt_event_counts=prompt_events,
            prompt_support=prompt_support,
            prompt_relay_count=relay_count,
            generic_ascii_class=ascii_class,
            scan=_empty_scan(len(spans)),
            projected_membership=empty_bool,
            cell_ids=cell_ids,
            currents=(0.0,) * factor_checkpoint.FACTOR_COUNT,
            spikes=empty_bool,
        )
        return PromptFactorAttentionHandoff(_issue_attention_receipt(state))
    scan = _scan_factor_carriers(spans)
    rotation = intervention.factor_rotation % factor_checkpoint.FACTOR_COUNT
    projected = tuple(
        scan.physical_membership[(index + rotation) % factor_checkpoint.FACTOR_COUNT]
        for index in range(factor_checkpoint.FACTOR_COUNT)
    )
    if intervention.collide_cell_ids:
        cell_ids = tuple(index // 2 for index in range(factor_checkpoint.FACTOR_COUNT))
    collision_count = prompt_collisions + (len(cell_ids) - len(set(cell_ids)))
    if scan.status != SCAN_READY or collision_count:
        status = STOP_CAP if scan.status == STOP_CAP else STOP_COLLISION
        reason = (
            "factor carrier scan exceeded physical membership capacity"
            if status == STOP_CAP
            else "factor attention physical identities collided"
        )
        state = _build_state(
            status=status,
            stop_reason=reason,
            intervention=intervention,
            prompt=prompt,
            connector=connector,
            prompt_event_counts=prompt_events,
            prompt_support=prompt_support,
            prompt_relay_count=relay_count,
            generic_ascii_class=ascii_class,
            scan=scan,
            projected_membership=projected,
            cell_ids=cell_ids,
            currents=(0.0,) * factor_checkpoint.FACTOR_COUNT,
            spikes=empty_bool,
        )
        return PromptFactorAttentionHandoff(_issue_attention_receipt(state))

    currents_array = (
        np.asarray(prompt_support, dtype=np.float64) * PROMPT_CURRENT_PA
        + np.asarray(projected, dtype=np.float64) * FACTOR_CURRENT_PA
    ) * intervention.cell_gain
    population = _DiscreteNeuronPopulation(
        factor_checkpoint.FACTOR_COUNT,
        threshold_pa=COINCIDENCE_THRESHOLD_PA,
    )
    spike_array = population.step(currents_array)
    spikes = tuple(bool(value) for value in spike_array)
    membership_count = sum(spikes)
    if intervention.name == _CELL_OFF.name:
        status, reason = STOP_CELL_OFF, "coincidence cells were disabled"
    elif intervention.name == _DERANGED.name:
        status, reason = STOP_DERANGEMENT, "factor membership was deranged"
    elif membership_count == 0:
        status, reason = STOP_ZERO, "no prompt-supported factor membership spiked"
    elif membership_count > FACTOR_CAPACITY:
        status, reason = STOP_CAP, "prompt-supported factor membership exceeded capacity four"
    else:
        status, reason = ATTENTION_READY, None
    state = _build_state(
        status=status,
        stop_reason=reason,
        intervention=intervention,
        prompt=prompt,
        connector=connector,
        prompt_event_counts=prompt_events,
        prompt_support=prompt_support,
        prompt_relay_count=relay_count,
        generic_ascii_class=ascii_class,
        scan=scan,
        projected_membership=projected,
        cell_ids=cell_ids,
        currents=tuple(float(value) for value in currents_array),
        spikes=spikes,
    )
    return PromptFactorAttentionHandoff(_issue_attention_receipt(state))


def infer_prompt_factor_attention(
    span_records: Sequence[Mapping[str, object]],
    literal_records: Sequence[Mapping[str, object]],
    signature_role_record: Mapping[str, object],
) -> PromptFactorAttentionHandoff:
    """Run treatment attention with no caller-provided factor observation."""

    return _run_attention(
        span_records,
        literal_records,
        signature_role_record,
        _TREATMENT,
    )


def require_prompt_factor_attention(
    handoff: PromptFactorAttentionHandoff,
) -> PromptFactorAttentionState:
    """Authenticate a successful attention receipt for the future joint seam."""

    if type(handoff) is not PromptFactorAttentionHandoff:
        raise TypeError("attention authority requires its opaque handoff")
    receipt = handoff._receipt
    if not _authenticate_attention_receipt(receipt):
        raise RuntimeError("prompt factor attention receipt authentication failed")
    if receipt.state.status != ATTENTION_READY:
        raise RuntimeError(receipt.state.stop_reason or "prompt factor attention stopped")
    return receipt.state


__all__ = [
    "ATTENTION_READY",
    "FACTOR_CAPACITY",
    "SCAN_READY",
    "SCHEMA",
    "SELECTOR_GEOMETRY_COUNT",
    "STOP_CAP",
    "STOP_CELL_OFF",
    "STOP_COLLISION",
    "STOP_CONNECTOR",
    "STOP_DERANGEMENT",
    "STOP_PROMPT",
    "STOP_ZERO",
    "FactorCarrierScanAudit",
    "PromptFactorAttentionHandoff",
    "PromptFactorAttentionState",
    "infer_prompt_factor_attention",
    "require_prompt_factor_attention",
]
