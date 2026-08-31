"""Exact bound execution for the fresh public-TRAIN V2 joint roster.

The protocol owns the opaque rows and desired-free payloads.  This module owns
the physical batch producer, static occurrence inference, converter, causal
interventions, and an exact payload-to-frontier binding.  It deliberately has
no desired-opening API; the one-shot runner may select only after authenticating
one of the bindings issued here.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from typing import NamedTuple

from experiments.r3_pilot import (
    public_train_fit_occurrence_factor_static_checkpoint_capsule_v1 as occurrence_static,
)
from experiments.r3_pilot import (
    public_train_fit_prompt_record_canonical_joint_v2_protocol as protocol,
)
from experiments.r3_pilot import (
    public_train_fit_prompt_record_source_event_batch_selector_v1 as batch_producer,
)
from experiments.r3_pilot import (
    public_train_fit_prompt_record_source_event_producer_v1 as prompt_producer,
)
from experiments.r3_pilot import (
    typed_semantic_packed_public_train_fit_occurrence_factor_authorized_plan_slice_v1,
)

packed_converter = (
    typed_semantic_packed_public_train_fit_occurrence_factor_authorized_plan_slice_v1
)

SCHEMA = "PUBLIC_TRAIN_FIT_PROMPT_RECORD_CANONICAL_JOINT_V2_BOUND_EXECUTION"
BOUND_READY = "PUBLIC_TRAIN_FIT_PROMPT_RECORD_CANONICAL_JOINT_V2_BOUND_READY"
ZERO_SELECTED_SOURCE_EVENTS = "ZERO_SELECTED_SOURCE_EVENTS"
STOP_BOUND_EXECUTION = (
    "STOP_PUBLIC_TRAIN_FIT_PROMPT_RECORD_CANONICAL_JOINT_V2_BOUND_EXECUTION"
)

MODE_PRODUCER_OFF = "producer-off"
MODE_STATIC_OFF = "static-cells-off"
MODE_CONVERTER_OFF = "converter-authority-off"
MODE_WRONG_DESIRED = "desired-permutation"
MODE_NORMAL = "normal"
TREATMENT_ORDER = (
    MODE_PRODUCER_OFF,
    MODE_STATIC_OFF,
    MODE_CONVERTER_OFF,
    MODE_WRONG_DESIRED,
    MODE_NORMAL,
)

PROTOCOL_BANK_COMMIT = "61477f9797dbe6df6b6e7f0653e701c5660fb03e"
BATCH_PRODUCER_BANK_COMMIT = "a3a42912812c51965fcfa8be7706eb002a4b6872"
PROMPT_PRODUCER_BANK_COMMIT = "1232d2ddc14501557ac8d85e8c0846a7c4a7cee2"
STATIC_BANK_COMMIT = "d2376c7dc75314ae285ffb98edeae3e704b63d49"
CONVERTER_BANK_COMMIT = "41ad28ca7f136077a1e677029864178837b7d34d"
PLAN_SLICE_BANK_COMMIT = "41cc0a555ffb5e0fe51e8c9c408e9ac914fb491d"

PROTOCOL_SOURCE_SHA256 = (
    "68bdbf8a36d77b01d204dfeaec46f7cabb44f45ffcdca272c2a481b58cb71670"
)
BATCH_PRODUCER_SOURCE_SHA256 = (
    "54ff8bcc8ef4df63104e54df8113d39473bb00eacec409c0bbf8065c526e0074"
)
PROMPT_PRODUCER_SOURCE_SHA256 = (
    "5e195bccf8c8ebbe59bfc6d89f6f27b9fc15ee6fe8a6a62cbe96f57650d04784"
)
STATIC_SOURCE_SHA256 = (
    "0af68609e46afd65e8dc50c656f7ad22919a08329045d3068f8a457813870c89"
)
CONVERTER_SOURCE_SHA256 = (
    "daa8a119caec7c2700c46fadb569676fed7742413899856de6d6e2d12bf5a90c"
)
PLAN_SLICE_SOURCE_SHA256 = (
    "012cd2a52794a74d966223d50c1e7c645e5aeb150f4326bde531c8cf81f29576"
)

PROTOCOL_ROOT_SHA256 = (
    "5eb01df63fa0d0f375e5180bde91d01a395e509f2f34e6a46782fed77d53ea91"
)
PROTOCOL_DEPENDENCY_SHA256 = (
    "1adf8e2b9776a763855518b417802035490ce6bc9aff0c354fdc22a1235d6044"
)
PROTOCOL_OUTCOME_ROOT_SHA256 = (
    "5c347269720a4ed0623241849ba6c36be23d801b848b8859c909f00818668e5a"
)
PROTOCOL_WORK_ROOT_SHA256 = (
    "d39ec037b62728064a3db154f8c6ff7a1c613fcbebf60ca972c78758dce74b45"
)

READY = packed_converter.READY
SELECTED = packed_converter.SELECTED
STOP_SOURCE_CERTIFICATE = packed_converter.STOP_SOURCE_CERTIFICATE
STOP_NO_EXACT = packed_converter.plan_slice.STOP_NO_EXACT
STOP_AUTHORIZED_ECLASS_MISMATCH = (
    packed_converter.plan_slice.STOP_AUTHORIZED_ECLASS_MISMATCH
)

REPO_ROOT = Path(__file__).resolve().parents[2]
_NAMESPACE = b"public-train-fit-prompt-record-canonical-joint-v2-bound-execution\0"
_DEPENDENCY_SHA256S = {
    "experiments/r3_pilot/public_train_fit_prompt_record_canonical_joint_v2_protocol.py": (
        PROTOCOL_SOURCE_SHA256
    ),
    "experiments/r3_pilot/public_train_fit_prompt_record_source_event_batch_selector_v1.py": (
        BATCH_PRODUCER_SOURCE_SHA256
    ),
    "experiments/r3_pilot/public_train_fit_prompt_record_source_event_producer_v1.py": (
        PROMPT_PRODUCER_SOURCE_SHA256
    ),
    "experiments/r3_pilot/public_train_fit_occurrence_factor_static_checkpoint_capsule_v1.py": (
        STATIC_SOURCE_SHA256
    ),
    (
        "experiments/r3_pilot/typed_semantic_packed_public_train_fit_"
        "occurrence_factor_authorized_plan_slice_v1.py"
    ): CONVERTER_SOURCE_SHA256,
    "experiments/r3_pilot/typed_semantic_packed_prompt_authorized_plan_slice_v1.py": (
        PLAN_SLICE_SOURCE_SHA256
    ),
}


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(_plain(item) for item in value)
    if hasattr(value, "to_record"):
        return _plain(value.to_record())
    if hasattr(value, "_asdict"):
        return _plain(value._asdict())
    return value


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        _plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _digest(label: str, value: object) -> str:
    result = sha256()
    result.update(_NAMESPACE)
    result.update(label.encode("utf-8"))
    result.update(b"\0")
    result.update(_canonical_json(value))
    return result.hexdigest()


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def authenticate_dependencies() -> str:
    """Fail closed if any banked physical execution dependency changed."""

    for relative, expected in _DEPENDENCY_SHA256S.items():
        if _file_sha256(REPO_ROOT / relative) != expected:
            raise BoundExecutionStop(
                STOP_BOUND_EXECUTION, f"banked bound-execution dependency changed: {relative}"
            )
    frozen = protocol.PROTOCOL
    if (
        frozen.protocol_sha256 != PROTOCOL_ROOT_SHA256
        or frozen.dependency_sha256 != PROTOCOL_DEPENDENCY_SHA256
        or frozen.outcome_contract_sha256 != PROTOCOL_OUTCOME_ROOT_SHA256
        or frozen.work_formula_sha256 != PROTOCOL_WORK_ROOT_SHA256
        or protocol.WORK_FORMULA.fitted_glif_steps_per_query
        != prompt_producer._OUTPUT_STEPS
        or protocol.WORK_FORMULA.fitted_glif_steps_per_query
        != occurrence_static.OUTPUT_STEPS
        or tuple(item.name for item in protocol.CONTROL_CONTRACTS)
        != TREATMENT_ORDER[:-1]
    ):
        raise BoundExecutionStop(
            STOP_BOUND_EXECUTION, "banked V2 execution contract changed"
        )
    return _digest(
        "dependencies",
        {
            "files": _DEPENDENCY_SHA256S,
            "protocol_root_sha256": PROTOCOL_ROOT_SHA256,
            "protocol_dependency_sha256": PROTOCOL_DEPENDENCY_SHA256,
        },
    )


class BoundExecutionStop(RuntimeError):
    """Fail-closed exact binding or physical execution error."""

    def __init__(self, status: str, reason: str) -> None:
        self.status = status
        super().__init__(reason)


class _BindingReceipt(NamedTuple):
    preflight: protocol.FrozenPairingPreflight
    row: protocol.FrozenPublicTrainJointRow
    payload: protocol.FrozenJointBuildPayload
    geometry: protocol.CandidateGeometryReceipt
    source_certificate: prompt_producer.PromptRecordSourceEventCertificate
    source_receipt: object
    internal_static_certificate: object | None
    internal_static_view: object | None
    explicit_static_certificate: object | None
    explicit_static_view: object | None
    frontier: object | None
    mode: str
    row_ordinal: int
    dependency_sha256: str
    runtime_fields_sha256: str
    payload_sha256: str
    frontier_sha256: str | None
    build_status: str
    occurrence_status: str | None
    source_event_count: int
    occurrence_count: int
    occurrence_inference_attempts: int
    plan_build_attempts: int
    static_query_count: int
    sha256: str


class ExactPayloadFrontierBinding:
    """Opaque exact payload/certificate/frontier authority."""

    __slots__ = ()

    def __new__(cls) -> ExactPayloadFrontierBinding:
        raise TypeError("exact V2 execution bindings require private authority")

    @property
    def mode(self) -> str:
        return _lookup_binding(self).mode

    @property
    def row_ordinal(self) -> int:
        return _lookup_binding(self).row_ordinal

    @property
    def status(self) -> str:
        return _lookup_binding(self).build_status

    @property
    def requires_desired(self) -> bool:
        receipt = _lookup_binding(self)
        return receipt.mode in (MODE_WRONG_DESIRED, MODE_NORMAL) and (
            receipt.build_status == READY
        )

    @property
    def sha256(self) -> str:
        return _lookup_binding(self).sha256


class TreatmentOutcome(NamedTuple):
    mode: str
    row_ordinal: int
    build_status: str
    source_event_status: str | None
    occurrence_status: str | None
    public_status: str
    selection_status: str | None
    source_candidate_event_count: int
    source_event_count: int
    occurrence_count: int
    producer_build_attempts: int
    occurrence_inference_attempts: int
    plan_build_attempts: int
    runtime_desired_reads: int
    exact_proof_count: int
    witness_count: int
    selected_proof_forest_sha256: str | None
    relation_work_units: int
    physical_predictions: int
    physical_races: int
    static_query_count: int
    payload_sha256: str
    geometry_sha256: str
    source_certificate_sha256: str
    source_bundle_sha256: str | None
    static_certificate_sha256s: tuple[str, ...]
    frontier_sha256: str | None
    binding_sha256: str
    work_sha256: str | None
    sha256: str


_ISSUED_BINDINGS: dict[
    int, tuple[ExactPayloadFrontierBinding, _BindingReceipt]
] = {}
_ISSUED_OUTCOMES: dict[int, TreatmentOutcome] = {}
_INTERVENTION_ACTIVE = False


def _runtime_fields(payload: protocol.FrozenJointBuildPayload) -> dict[str, object]:
    return {
        "span_records": payload.span_records,
        "literal_records": payload.literal_records,
        "signature_role_record": payload.signature_role_record,
        "signature": payload.signature,
        "cases": payload.cases,
        "literals": payload.literals,
        "max_cost": payload.max_cost,
        "desired_present": False,
    }


def _runtime_fields_sha256(payload: protocol.FrozenJointBuildPayload) -> str:
    return _digest("runtime-build-fields", _runtime_fields(payload))


def _binding_payload(receipt: _BindingReceipt) -> dict[str, object]:
    source = receipt.source_receipt
    return {
        "preflight_sha256": receipt.preflight.sha256,
        "row_pairing_sha256": protocol._lookup_row(receipt.row).pairing_sha256,
        "payload_sha256": receipt.payload_sha256,
        "geometry_sha256": receipt.geometry.sha256,
        "source_certificate_sha256": receipt.source_certificate.sha256,
        "source_request_sha256": source.request_sha256,
        "source_bundle_sha256": source.bundle_sha256,
        "internal_static_certificate_sha256": (
            None
            if receipt.internal_static_certificate is None
            else receipt.internal_static_certificate.sha256
        ),
        "explicit_static_certificate_sha256": (
            None
            if receipt.explicit_static_certificate is None
            else receipt.explicit_static_certificate.sha256
        ),
        "frontier_sha256": receipt.frontier_sha256,
        "mode": receipt.mode,
        "row_ordinal": receipt.row_ordinal,
        "dependency_sha256": receipt.dependency_sha256,
        "runtime_fields_sha256": receipt.runtime_fields_sha256,
        "build_status": receipt.build_status,
        "occurrence_status": receipt.occurrence_status,
        "source_event_count": receipt.source_event_count,
        "occurrence_count": receipt.occurrence_count,
        "occurrence_inference_attempts": receipt.occurrence_inference_attempts,
        "plan_build_attempts": receipt.plan_build_attempts,
        "static_query_count": receipt.static_query_count,
        "exact_object_binding": True,
    }


def _issue_binding(receipt: _BindingReceipt) -> ExactPayloadFrontierBinding:
    binding = object.__new__(ExactPayloadFrontierBinding)
    _ISSUED_BINDINGS[id(binding)] = (binding, receipt)
    return binding


def _lookup_binding(binding: object) -> _BindingReceipt:
    if type(binding) is not ExactPayloadFrontierBinding:
        raise BoundExecutionStop(
            protocol.STOP_PROMPT_IO_PAIRING_MISMATCH,
            "payload-frontier binding has the wrong type",
        )
    issued = _ISSUED_BINDINGS.get(id(binding))
    if issued is None or issued[0] is not binding:
        raise BoundExecutionStop(
            protocol.STOP_PROMPT_IO_PAIRING_MISMATCH,
            "payload-frontier binding is not an issued exact object",
        )
    receipt = issued[1]
    access = protocol._lookup_build_access(receipt.payload)
    row_receipt = protocol._lookup_row(receipt.row)
    if (
        access.preflight is not receipt.preflight
        or access.row is not receipt.row
        or access.build_sha256 != receipt.payload_sha256
        or row_receipt.geometry is not receipt.geometry
        or receipt.geometry is not protocol.CANDIDATE_GEOMETRY_RECEIPTS[
            receipt.row_ordinal
        ]
        or _runtime_fields_sha256(receipt.payload) != receipt.runtime_fields_sha256
    ):
        raise BoundExecutionStop(
            protocol.STOP_PROMPT_IO_PAIRING_MISMATCH,
            "protocol payload or candidate geometry binding changed",
        )
    source_receipt = prompt_producer._authenticate_certificate(
        receipt.source_certificate
    )
    if source_receipt is not receipt.source_receipt:
        raise BoundExecutionStop(
            protocol.STOP_PROMPT_IO_PAIRING_MISMATCH,
            "source certificate binding changed",
        )
    for certificate, view in (
        (receipt.internal_static_certificate, receipt.internal_static_view),
        (receipt.explicit_static_certificate, receipt.explicit_static_view),
    ):
        if certificate is None:
            if view is not None:
                raise BoundExecutionStop(
                    protocol.STOP_PROMPT_IO_PAIRING_MISMATCH,
                    "static view escaped without its exact certificate",
                )
            continue
        static_receipt = occurrence_static._authenticate_certificate(certificate)
        if view is not None and static_receipt.view is not view:
            raise BoundExecutionStop(
                protocol.STOP_PROMPT_IO_PAIRING_MISMATCH,
                "static certificate/view binding changed",
            )
    if receipt.frontier is not None:
        receipt.frontier._authenticate()
        if receipt.frontier.sha256 != receipt.frontier_sha256:
            raise BoundExecutionStop(
                protocol.STOP_PROMPT_IO_PAIRING_MISMATCH,
                "converter frontier changed after exact binding",
            )
    if receipt.sha256 != _digest("exact-binding", _binding_payload(receipt)):
        raise BoundExecutionStop(
            protocol.STOP_PROMPT_IO_PAIRING_MISMATCH,
            "exact binding receipt changed",
        )
    return receipt


def authenticate_binding(binding: ExactPayloadFrontierBinding) -> _BindingReceipt:
    """Authenticate and return the private receipt for runner-owned selection."""

    return _lookup_binding(binding)


@contextmanager
def _static_cells_off():
    global _INTERVENTION_ACTIVE
    if _INTERVENTION_ACTIVE:
        raise RuntimeError("V2 physical interventions cannot overlap")
    _INTERVENTION_ACTIVE = True
    original = occurrence_static._decode

    def cell_off(query: object, *, cell_gain: float = 1.0) -> object:
        del cell_gain
        return original(query, cell_gain=0.0)

    occurrence_static._decode = cell_off
    try:
        yield
    finally:
        occurrence_static._decode = original
        _INTERVENTION_ACTIVE = False


def _complete_static_queries(source_count: int) -> int:
    return source_count * source_count + 18 * source_count


def _forge_unissued_static_certificate(
    certificate: occurrence_static.OccurrenceFactorStaticCertificate,
) -> occurrence_static.OccurrenceFactorStaticCertificate:
    occurrence_static._authenticate_certificate(certificate)
    forged = object.__new__(occurrence_static.OccurrenceFactorStaticCertificate)
    forged._receipt = certificate._receipt
    if occurrence_static._ISSUED_CERTIFICATES.get(id(forged)) is not None:
        raise RuntimeError("converter control unexpectedly issued its forged certificate")
    return forged


def bind_treatment(
    preflight: protocol.FrozenPairingPreflight,
    row: protocol.FrozenPublicTrainJointRow,
    row_ordinal: int,
    mode: str,
) -> ExactPayloadFrontierBinding:
    """Run one desired-free build and bind every exact physical object used."""

    if mode not in TREATMENT_ORDER:
        raise ValueError(f"unknown frozen V2 treatment mode: {mode}")
    if type(row_ordinal) is not int or not 0 <= row_ordinal < len(protocol.ROWS):
        raise ValueError("row ordinal is outside the frozen V2 roster")
    if protocol.ROWS[row_ordinal] is not row:
        raise BoundExecutionStop(
            protocol.STOP_PROMPT_IO_PAIRING_MISMATCH,
            "row ordinal substituted a different exact V2 row",
        )
    dependency_sha = authenticate_dependencies()
    payload = protocol.require_build_payload(preflight, row)
    access = protocol._lookup_build_access(payload)
    if access.preflight is not preflight or access.row is not row:
        raise BoundExecutionStop(
            protocol.STOP_PROMPT_IO_PAIRING_MISMATCH,
            "protocol issued a payload outside its exact row preflight",
        )
    geometry = protocol.CANDIDATE_GEOMETRY_RECEIPTS[row_ordinal]
    if protocol._lookup_row(row).geometry is not geometry:
        raise BoundExecutionStop(
            protocol.STOP_PROMPT_IO_PAIRING_MISMATCH,
            "row candidate geometry changed before execution",
        )
    runtime_sha = _runtime_fields_sha256(payload)

    if mode == MODE_PRODUCER_OFF:
        source_certificate = batch_producer.produce_source_event_bundle_batch(
            payload.span_records,
            payload.literal_records,
            payload.signature_role_record,
            cell_gain=0.0,
        )
    elif mode == MODE_STATIC_OFF:
        with _static_cells_off():
            source_certificate = batch_producer.produce_source_event_bundle_batch(
                payload.span_records,
                payload.literal_records,
                payload.signature_role_record,
                cell_gain=1.0,
            )
    else:
        source_certificate = batch_producer.produce_source_event_bundle_batch(
            payload.span_records,
            payload.literal_records,
            payload.signature_role_record,
            cell_gain=1.0,
        )

    source_receipt = prompt_producer._authenticate_certificate(source_certificate)
    if source_receipt.candidate_event_count != geometry.total_candidate_count:
        raise BoundExecutionStop(
            protocol.STOP_PROMPT_IO_PAIRING_MISMATCH,
            "batch producer candidate count differs from preregistered geometry",
        )

    internal_static = source_receipt.downstream_certificate
    internal_view = None
    explicit_static = None
    explicit_view = None
    frontier = None
    frontier_sha = None
    occurrence_status = None
    source_count = len(source_receipt.events)
    occurrence_count = 0
    occurrence_attempts = 0
    plan_attempts = 0
    static_queries = 0

    if mode == MODE_PRODUCER_OFF:
        if (
            source_receipt.status != prompt_producer.STOP_ZERO
            or source_count != 0
            or internal_static is not None
        ):
            raise BoundExecutionStop(
                STOP_BOUND_EXECUTION,
                "producer-off batch did not return authentic STOP_ZERO",
            )
        build_status = ZERO_SELECTED_SOURCE_EVENTS
    elif mode == MODE_STATIC_OFF:
        if internal_static is None:
            raise BoundExecutionStop(
                STOP_BOUND_EXECUTION,
                "static-off producer omitted its downstream STOP certificate",
            )
        internal_receipt = occurrence_static._authenticate_certificate(internal_static)
        occurrence_attempts = 1
        occurrence_status = internal_receipt.status
        static_queries = source_count
        if (
            source_receipt.status != prompt_producer.STOP_UNSUPPORTED
            or source_count not in protocol.SUPPORTED_SOURCE_EVENT_COUNTS
            or internal_receipt.status != occurrence_static.STOP_ZERO
            or internal_receipt.view is not None
        ):
            raise BoundExecutionStop(
                STOP_BOUND_EXECUTION,
                "static-cells-off did not causally stop the downstream inference",
            )
        build_status = occurrence_static.STOP_ZERO
    else:
        if (
            source_receipt.status != prompt_producer.READY
            or internal_static is None
            or source_count not in protocol.SUPPORTED_SOURCE_EVENT_COUNTS
        ):
            raise BoundExecutionStop(
                STOP_BOUND_EXECUTION,
                "batch producer did not issue one supported READY source bundle",
            )
        events = prompt_producer.require_source_event_bundle(source_certificate)
        internal_view = occurrence_static.require_occurrence_factors(internal_static)
        explicit_static = occurrence_static.infer_occurrence_factors(events)
        explicit_view = occurrence_static.require_occurrence_factors(explicit_static)
        occurrence_attempts = 2
        occurrence_status = explicit_static.status
        occurrence_count = len(explicit_view.occurrence_carriers)
        static_queries = 2 * _complete_static_queries(source_count)
        if occurrence_count != source_count:
            raise BoundExecutionStop(
                STOP_BOUND_EXECUTION,
                "explicit static occurrence count differs from the source bundle",
            )
        converter_source = explicit_static
        if mode == MODE_CONVERTER_OFF:
            converter_source = _forge_unissued_static_certificate(explicit_static)
        frontier = (
            packed_converter.PackedPublicTrainFitOccurrenceFactorAuthorizedPlanSliceV1.build(
                payload.signature,
                payload.cases,
                payload.literals,
                payload.max_cost,
                converter_source,
            )
        )
        frontier._authenticate()
        build_status = frontier.status
        plan_attempts = frontier.state.plan_build_attempts
        frontier_sha = frontier.sha256
        if mode == MODE_CONVERTER_OFF:
            if build_status != STOP_SOURCE_CERTIFICATE or plan_attempts != 0:
                raise BoundExecutionStop(
                    STOP_BOUND_EXECUTION,
                    "converter authority control accepted an unissued certificate",
                )
        elif build_status != READY or plan_attempts != 1:
            raise BoundExecutionStop(
                STOP_BOUND_EXECUTION,
                "exact converter frontier did not become desired-free READY",
            )

    if _runtime_fields_sha256(payload) != runtime_sha:
        raise BoundExecutionStop(
            protocol.STOP_PROMPT_IO_PAIRING_MISMATCH,
            "runtime build fields changed during physical construction",
        )
    partial = _BindingReceipt(
        preflight,
        row,
        payload,
        geometry,
        source_certificate,
        source_receipt,
        internal_static,
        internal_view,
        explicit_static,
        explicit_view,
        frontier,
        mode,
        row_ordinal,
        dependency_sha,
        runtime_sha,
        payload.sha256,
        frontier_sha,
        build_status,
        occurrence_status,
        source_count,
        occurrence_count,
        occurrence_attempts,
        plan_attempts,
        static_queries,
        "",
    )
    receipt = partial._replace(
        sha256=_digest("exact-binding", _binding_payload(partial))
    )
    binding = _issue_binding(receipt)
    _lookup_binding(binding)
    return binding


def _work_values(work: object | None) -> tuple[int, int, int, int, str | None]:
    if work is None:
        return (0, 0, 0, 0, None)
    relation = getattr(work, "relation", None)
    values = (
        getattr(relation, "completed_units", None),
        getattr(work, "physical_predictions", None),
        getattr(work, "physical_races", None),
        getattr(work, "concrete_proof_witnesses", None),
    )
    work_sha = getattr(work, "sha256", None)
    if not all(type(value) is int and value >= 0 for value in values):
        raise TypeError("packed frontier work audit is incomplete")
    if type(work_sha) is not str or len(work_sha) != 64:
        raise TypeError("packed frontier work audit has no exact SHA-256 receipt")
    return (*values, work_sha)


def _static_certificate_sha256s(receipt: _BindingReceipt) -> tuple[str, ...]:
    return tuple(
        certificate.sha256
        for certificate in (
            receipt.internal_static_certificate,
            receipt.explicit_static_certificate,
        )
        if certificate is not None
    )


def _issue_outcome(
    receipt: _BindingReceipt,
    *,
    public_status: str,
    selection_status: str | None,
    runtime_desired_reads: int,
    exact_proof_count: int,
    witness_count: int,
    selected_proof_forest_sha256: str | None,
    relation_work_units: int,
    physical_predictions: int,
    physical_races: int,
    work_sha256: str | None,
) -> TreatmentOutcome:
    source = receipt.source_receipt
    values = (
        receipt.mode,
        receipt.row_ordinal,
        receipt.build_status,
        source.status,
        receipt.occurrence_status,
        public_status,
        selection_status,
        source.candidate_event_count,
        receipt.source_event_count,
        receipt.occurrence_count,
        1,
        receipt.occurrence_inference_attempts,
        receipt.plan_build_attempts,
        runtime_desired_reads,
        exact_proof_count,
        witness_count,
        selected_proof_forest_sha256,
        relation_work_units,
        physical_predictions,
        physical_races,
        receipt.static_query_count,
        receipt.payload_sha256,
        receipt.geometry.sha256,
        receipt.source_certificate.sha256,
        source.bundle_sha256,
        _static_certificate_sha256s(receipt),
        receipt.frontier_sha256,
        receipt.sha256,
        work_sha256,
    )
    outcome = TreatmentOutcome(*values, _digest("treatment-outcome", values))
    _ISSUED_OUTCOMES[id(outcome)] = outcome
    return outcome


def finish_bound(binding: ExactPayloadFrontierBinding) -> TreatmentOutcome:
    """Publish one causal-control outcome that requires no desired opening."""

    receipt = _lookup_binding(binding)
    if binding.requires_desired:
        raise BoundExecutionStop(
            STOP_BOUND_EXECUTION,
            "READY payload-frontier binding requires runner-owned desired selection",
        )
    work = None if receipt.frontier is None else receipt.frontier.work
    relation, predictions, races, work_proofs, work_sha = _work_values(work)
    if work_proofs != 0:
        raise BoundExecutionStop(
            STOP_BOUND_EXECUTION, "desired-free control materialized proof witnesses"
        )
    return _issue_outcome(
        receipt,
        public_status=receipt.build_status,
        selection_status=None,
        runtime_desired_reads=0,
        exact_proof_count=0,
        witness_count=0,
        selected_proof_forest_sha256=None,
        relation_work_units=relation,
        physical_predictions=predictions,
        physical_races=races,
        work_sha256=work_sha,
    )


def select_bound(
    binding: ExactPayloadFrontierBinding,
    desired: Sequence[object],
) -> TreatmentOutcome:
    """Select only through an authenticated exact READY binding."""

    receipt = _lookup_binding(binding)
    if not binding.requires_desired or receipt.frontier is None:
        raise BoundExecutionStop(
            STOP_BOUND_EXECUTION, "binding is not eligible for desired selection"
        )
    frozen_desired = tuple(desired)
    selection = receipt.frontier.select(frozen_desired)
    _lookup_binding(binding)
    if selection.build_state_sha256 != receipt.frontier_sha256:
        raise BoundExecutionStop(
            protocol.STOP_PROMPT_IO_PAIRING_MISMATCH,
            "selection is not bound to the exact executed frontier",
        )
    selection_status = selection.status
    public_status = selection_status
    if selection_status == STOP_AUTHORIZED_ECLASS_MISMATCH:
        if (
            selection.exact_proof_count != 0
            or tuple(selection.witnesses)
            or selection.selected_proof_forest_sha256 is not None
        ):
            raise BoundExecutionStop(
                protocol.STOP_PROMPT_IO_PAIRING_MISMATCH,
                "authorized mismatch exposed forbidden proof evidence",
            )
        public_status = protocol.STOP_PROMPT_IO_PAIRING_MISMATCH
    witnesses = tuple(selection.witnesses)
    relation, predictions, races, work_proofs, work_sha = _work_values(
        selection.work
    )
    if work_proofs != len(witnesses):
        raise BoundExecutionStop(
            STOP_BOUND_EXECUTION, "work proof count differs from published witnesses"
        )
    return _issue_outcome(
        receipt,
        public_status=public_status,
        selection_status=selection_status,
        runtime_desired_reads=len(frozen_desired),
        exact_proof_count=selection.exact_proof_count,
        witness_count=len(witnesses),
        selected_proof_forest_sha256=selection.selected_proof_forest_sha256,
        relation_work_units=relation,
        physical_predictions=predictions,
        physical_races=races,
        work_sha256=work_sha,
    )


def authenticate_outcome(outcome: TreatmentOutcome) -> TreatmentOutcome:
    """Authenticate an exact outcome issued by this execution engine."""

    if type(outcome) is not TreatmentOutcome or _ISSUED_OUTCOMES.get(id(outcome)) is not outcome:
        raise TypeError("treatment outcome is not an issued exact object")
    if outcome.sha256 != _digest("treatment-outcome", outcome[:-1]):
        raise RuntimeError("treatment outcome receipt changed")
    return outcome


CLAIM_BOUNDARY = (
    "fresh two-row public-TRAIN same-FIT exact physical batch/static/converter "
    "binding and causal controls only; desired values open only in the separate "
    "one-shot runner, and unseen prompt/family transfer, language understanding, "
    "coding capability, DEV, heldout, M0, C1, G0, and protected evidence remain "
    "unproven"
)


__all__ = [
    "BOUND_READY",
    "CLAIM_BOUNDARY",
    "MODE_CONVERTER_OFF",
    "MODE_NORMAL",
    "MODE_PRODUCER_OFF",
    "MODE_STATIC_OFF",
    "MODE_WRONG_DESIRED",
    "SCHEMA",
    "STOP_BOUND_EXECUTION",
    "TREATMENT_ORDER",
    "ExactPayloadFrontierBinding",
    "TreatmentOutcome",
    "authenticate_binding",
    "authenticate_dependencies",
    "authenticate_outcome",
    "bind_treatment",
    "finish_bound",
    "select_bound",
]
