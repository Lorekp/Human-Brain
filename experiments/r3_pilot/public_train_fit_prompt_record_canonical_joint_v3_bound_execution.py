"""Exact desired-free execution for the fresh public-TRAIN V3 joint roster.

The protocol owns exact rows and desired-free payloads.  This module executes
the frozen two-pass source selector, reuses the one gain-1 source certificate
for both static interventions, and opens the packed converter only after an
authentic static READY certificate.  Desired values are accepted only by the
separate selection seam intended for an exclusive one-shot runner.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path
from typing import NamedTuple

from experiments.r3_pilot import (
    public_train_fit_occurrence_factor_static_checkpoint_capsule_v1 as occurrence_static,
)
from experiments.r3_pilot import (
    public_train_fit_prompt_record_canonical_joint_v3_protocol as protocol,
)
from experiments.r3_pilot import (
    public_train_fit_prompt_record_source_static_split_v1 as source_static,
)
from experiments.r3_pilot import (
    typed_semantic_packed_public_train_fit_occurrence_factor_authorized_plan_slice_v1,
)

packed_converter = (
    typed_semantic_packed_public_train_fit_occurrence_factor_authorized_plan_slice_v1
)

SCHEMA = "PUBLIC_TRAIN_FIT_PROMPT_RECORD_CANONICAL_JOINT_V3_BOUND_EXECUTION"
STOP_BOUND_EXECUTION = (
    "STOP_PUBLIC_TRAIN_FIT_PROMPT_RECORD_CANONICAL_JOINT_V3_BOUND_EXECUTION"
)
MODE_WRONG_DESIRED = "wrong-desired"
MODE_NORMAL = "normal"
SELECTION_ORDER = (MODE_WRONG_DESIRED, MODE_NORMAL)

PROTOCOL_BANK_COMMIT = "0541ee2adb37bfed7a50f2b484859f9fc35e946a"
SOURCE_STATIC_SPLIT_BANK_COMMIT = "5a643cae644e891f15a8c6fb8efdeadb1a612b7b"
STATIC_BANK_COMMIT = "d2376c7dc75314ae285ffb98edeae3e704b63d49"
CONVERTER_BANK_COMMIT = "41ad28ca7f136077a1e677029864178837b7d34d"
PLAN_SLICE_BANK_COMMIT = "41cc0a555ffb5e0fe51e8c9c408e9ac914fb491d"

PROTOCOL_SOURCE_SHA256 = (
    "6d38b9159a9b1634ce085c1a8145c7f5ff3e3c550cac36258f551fcb2a359935"
)
SOURCE_STATIC_SPLIT_SOURCE_SHA256 = (
    "9f56d3f3f6a3a499118c6ba2d6a9f6ad7ab0a3de195b6f2d77d64d05e271cc68"
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
    "265df3427f54bf95c1516bcf91081a6a54fef47b1b4e70bd7af8a92783d7d660"
)

READY = packed_converter.READY
SELECTED = packed_converter.SELECTED
STOP_SOURCE_CERTIFICATE = packed_converter.STOP_SOURCE_CERTIFICATE
STOP_AUTHORIZED_ECLASS_MISMATCH = (
    packed_converter.plan_slice.STOP_AUTHORIZED_ECLASS_MISMATCH
)

REPO_ROOT = Path(__file__).resolve().parents[2]
_NAMESPACE = b"public-train-fit-prompt-record-canonical-joint-v3-bound-execution\0"
_DEPENDENCY_SHA256S = {
    "experiments/r3_pilot/public_train_fit_prompt_record_canonical_joint_v3_protocol.py": (
        PROTOCOL_SOURCE_SHA256
    ),
    "experiments/r3_pilot/public_train_fit_prompt_record_source_static_split_v1.py": (
        SOURCE_STATIC_SPLIT_SOURCE_SHA256
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

# These are the only exits before the banked static capsule evaluates its full
# generic query set.  The dependency hash above makes this exact-work mapping
# part of the execution contract.
_EARLY_STATIC_STOP_REASONS = frozenset(
    {
        "no physical source event was supplied",
        "a physical source event was duplicated",
        "one physical source activated conflicting carrier fields",
        "one physical source had no learned contextual carrier",
        "multiple physical sources activated one carrier",
    }
)


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


class BoundExecutionStop(RuntimeError):
    """Fail-closed exact authority, work, or ordering error."""

    def __init__(self, status: str, reason: str) -> None:
        self.status = status
        super().__init__(reason)


def authenticate_dependencies() -> str:
    """Authenticate the exact banked protocol and physical execution stack."""

    for relative, expected in _DEPENDENCY_SHA256S.items():
        if _file_sha256(REPO_ROOT / relative) != expected:
            raise BoundExecutionStop(
                STOP_BOUND_EXECUTION, f"banked V3 execution dependency changed: {relative}"
            )
    try:
        protocol._authenticate_dependencies()
        protocol._authenticate_terminal_cohorts()
    except (RuntimeError, TypeError, ValueError) as error:
        raise BoundExecutionStop(STOP_BOUND_EXECUTION, str(error)) from error

    contract = protocol.SOURCE_STATIC_EXECUTION_CONTRACT
    work = protocol.WORK_FORMULA
    if (
        protocol.PROTOCOL.protocol_sha256 != PROTOCOL_ROOT_SHA256
        or contract.dependency_schema != source_static.SCHEMA
        or contract.dependency_source_sha256 != SOURCE_STATIC_SPLIT_SOURCE_SHA256
        or contract.dependency_root_sha256 != source_static.DEPENDENCY_ROOT_SHA256
        or contract.selector_passes_per_row != 2
        or contract.selector_gains != (0.0, 1.0)
        or contract.expected_source_statuses
        != (source_static.SOURCE_ZERO, source_static.SOURCE_SELECTED)
        or contract.gain1_source_certificate_count_per_row != 1
        or contract.gain1_source_certificate_required_exact is not True
        or contract.static_passes_per_row != 2
        or contract.static_gains != (0.0, 1.0)
        or contract.static_input_selector_passes != (1, 1)
        or contract.same_gain1_certificate_identity_required is not True
        or contract.selection_rerun_for_static_allowed is not False
        or contract.retain_source_status_and_roots is not True
        or contract.retain_static_status_and_roots_even_on_stop is not True
        or contract.host_selected_count_admission_gate_present is not False
        or contract.authenticated_static_on_gates_downstream is not True
        or contract.gated_downstream_modes
        != ("converter-authority-off", MODE_WRONG_DESIRED, MODE_NORMAL)
        or work.per_row_authenticated_candidate_counts != (554, 554)
        or work.selector_passes_per_row != 2
        or work.static_passes_per_row != 2
        or work.per_row_selected_count_bounds != ((1, 554), (1, 554))
        or work.fitted_glif_steps_per_query != 30
        or work.host_selected_count_admission_gate_present is not False
        or work.cap_independent_of_desired_and_outcomes is not True
        or occurrence_static.OUTPUT_STEPS != 30
        or packed_converter.source is not occurrence_static
    ):
        raise BoundExecutionStop(
            STOP_BOUND_EXECUTION, "banked V3 source/static execution contract changed"
        )
    return _digest(
        "dependencies",
        {
            "files": _DEPENDENCY_SHA256S,
            "protocol_root_sha256": PROTOCOL_ROOT_SHA256,
            "source_static_dependency_root_sha256": (
                source_static.DEPENDENCY_ROOT_SHA256
            ),
        },
    )


class _SourceStaticStage(NamedTuple):
    source_gain0_certificate: source_static.SourceSelectionCertificate
    source_gain0_receipt: object
    source_gain1_certificate: source_static.SourceSelectionCertificate
    source_gain1_receipt: object
    static_off_certificate: occurrence_static.OccurrenceFactorStaticCertificate
    static_off_receipt: object
    static_on_certificate: occurrence_static.OccurrenceFactorStaticCertificate
    static_on_receipt: object
    candidate_event_count: int
    selected_event_count: int
    source_query_count: int
    source_fitted_glif_step_count: int
    source_compiled_kernel_call_count: int
    static_off_query_count: int
    static_on_query_count: int
    static_query_count: int
    static_fitted_glif_step_count: int
    static_work_sha256: str


def _static_queries_for_one_complete_pass(source_count: int) -> int:
    return source_count + 6 * (source_count * source_count + 18 * source_count)


def _exact_static_query_count(receipt: object, source_count: int) -> int:
    status = getattr(receipt, "status", None)
    reason = getattr(receipt, "stop_reason", None)
    view = getattr(receipt, "view", None)
    complete = _static_queries_for_one_complete_pass(source_count)
    if status == occurrence_static.READY:
        if view is None:
            raise BoundExecutionStop(
                STOP_BOUND_EXECUTION, "static READY certificate omitted its relay view"
            )
        return complete
    if view is not None:
        raise BoundExecutionStop(
            STOP_BOUND_EXECUTION, "static STOP certificate retained a READY relay view"
        )
    if status not in (
        occurrence_static.STOP_ZERO,
        occurrence_static.STOP_AMBIGUOUS,
        occurrence_static.STOP_UNSUPPORTED,
    ):
        raise BoundExecutionStop(
            STOP_BOUND_EXECUTION, "static certificate has an unknown authentic status"
        )
    return source_count if reason in _EARLY_STATIC_STOP_REASONS else complete


def _expected_static_request_sha256(source_receipt: object) -> str:
    ordered = occurrence_static._canonical_source_bundle(source_receipt.selected_events)
    return occurrence_static._digest(
        "inference-request", tuple(event.neural_record() for event in ordered)
    )


def _run_source_static_stage(
    span_records: Sequence[Mapping[str, object]],
    literal_records: Sequence[Mapping[str, object]],
    signature_role_record: Mapping[str, object],
    *,
    expected_candidate_count: int,
) -> _SourceStaticStage:
    """Execute two selector passes and two same-certificate static passes."""

    gain0 = source_static.produce_source_selection_batch(
        span_records, literal_records, signature_role_record, cell_gain=0.0
    )
    gain1 = source_static.produce_source_selection_batch(
        span_records, literal_records, signature_role_record, cell_gain=1.0
    )
    gain0_receipt = source_static._authenticate_source_selection(gain0)
    gain1_receipt = source_static._authenticate_source_selection(gain1)
    selected_count = len(gain1_receipt.selected_events)

    if (
        gain0.status != source_static.SOURCE_ZERO
        or gain0.cell_gain != 0.0
        or gain0.candidate_event_count != expected_candidate_count
        or gain0.selected_event_count != 0
        or gain1.status != source_static.SOURCE_SELECTED
        or gain1.cell_gain != 1.0
        or gain1.candidate_event_count != expected_candidate_count
        or not 1 <= selected_count <= expected_candidate_count
        or gain1.selected_event_count != selected_count
        or gain0.request_sha256 != gain1.request_sha256
        or gain0.candidate_event_root_sha256
        != gain1.candidate_event_root_sha256
        or gain0.selector_fitted_glif_step_count
        != expected_candidate_count * occurrence_static.OUTPUT_STEPS
        or gain1.selector_fitted_glif_step_count
        != expected_candidate_count * occurrence_static.OUTPUT_STEPS
        or gain0.selector_compiled_kernel_call_count != 1
        or gain1.selector_compiled_kernel_call_count != 1
        or gain0.static_inference_call_count != 0
        or gain1.static_inference_call_count != 0
    ):
        raise BoundExecutionStop(
            STOP_BOUND_EXECUTION, "two-pass source selection violated its exact contract"
        )

    # Both calls deliberately receive the identical exact gain-1 object.
    static_off = source_static.infer_static_from_source(gain1, 0.0)
    static_on = source_static.infer_static_from_source(gain1, 1.0)
    static_off_receipt = occurrence_static._authenticate_certificate(static_off)
    static_on_receipt = occurrence_static._authenticate_certificate(static_on)
    expected_request = _expected_static_request_sha256(gain1_receipt)
    if (
        static_off_receipt.request_sha256 != expected_request
        or static_on_receipt.request_sha256 != expected_request
        or static_off_receipt.status != occurrence_static.STOP_ZERO
        or static_off_receipt.view is not None
    ):
        raise BoundExecutionStop(
            STOP_BOUND_EXECUTION,
            "same-certificate static interventions changed request authority",
        )

    static_off_queries = _exact_static_query_count(
        static_off_receipt, selected_count
    )
    static_on_queries = _exact_static_query_count(static_on_receipt, selected_count)
    complete_per_pass = _static_queries_for_one_complete_pass(selected_count)
    if (
        not 0 < static_off_queries <= complete_per_pass
        or not 0 < static_on_queries <= complete_per_pass
    ):
        raise BoundExecutionStop(
            STOP_BOUND_EXECUTION, "static query count exceeded its authenticated-n bound"
        )

    source_queries = 2 * expected_candidate_count
    source_fitted_steps = source_queries * occurrence_static.OUTPUT_STEPS
    static_queries = static_off_queries + static_on_queries
    static_fitted_steps = static_queries * occurrence_static.OUTPUT_STEPS
    if (
        source_queries > protocol.WORK_FORMULA.source_query_cap
        or static_queries > protocol.WORK_FORMULA.static_query_cap
        or source_fitted_steps + static_fitted_steps
        > protocol.WORK_FORMULA.fitted_glif_step_cap
    ):
        raise BoundExecutionStop(
            STOP_BOUND_EXECUTION, "V3 physical work exceeded a preregistered cap"
        )
    static_work = {
        "gain1_source_certificate_sha256": gain1.sha256,
        "selected_event_count": selected_count,
        "static_off_certificate_sha256": static_off.sha256,
        "static_off_query_count": static_off_queries,
        "static_on_certificate_sha256": static_on.sha256,
        "static_on_query_count": static_on_queries,
        "fitted_glif_step_count": static_fitted_steps,
        "complete_query_bound_per_pass": complete_per_pass,
    }
    return _SourceStaticStage(
        gain0,
        gain0_receipt,
        gain1,
        gain1_receipt,
        static_off,
        static_off_receipt,
        static_on,
        static_on_receipt,
        expected_candidate_count,
        selected_count,
        source_queries,
        source_fitted_steps,
        2,
        static_off_queries,
        static_on_queries,
        static_queries,
        static_fitted_steps,
        _digest("static-work", static_work),
    )


class _FrontierWork(NamedTuple):
    relation_completed_units: int
    relation_attempted_units: int
    relation_limit: int
    physical_predictions: int
    physical_races: int
    concrete_proof_witnesses: int
    sha256: str | None


_ZERO_FRONTIER_WORK = _FrontierWork(0, 0, 0, 0, 0, 0, None)


def _frontier_work(work: object | None) -> _FrontierWork:
    if work is None:
        return _ZERO_FRONTIER_WORK
    relation = getattr(work, "relation", None)
    values = (
        getattr(relation, "completed_units", None),
        getattr(relation, "attempted_units", None),
        getattr(relation, "limit", None),
        getattr(work, "physical_predictions", None),
        getattr(work, "physical_races", None),
        getattr(work, "concrete_proof_witnesses", None),
    )
    work_sha = getattr(work, "sha256", None)
    if (
        not all(type(value) is int and value >= 0 for value in values)
        or values[0] > values[2]
        or type(work_sha) is not str
        or len(work_sha) != 64
    ):
        raise BoundExecutionStop(
            STOP_BOUND_EXECUTION, "packed frontier work receipt is incomplete"
        )
    return _FrontierWork(*values, work_sha)


class _BindingReceipt(NamedTuple):
    preflight: protocol.FrozenPairingPreflight
    row: protocol.FrozenPublicTrainJointRow
    payload: protocol.FrozenJointBuildPayload
    geometry: protocol.CandidateGeometryReceipt
    stage: _SourceStaticStage
    converter_control_frontier: object | None
    frontier: object | None
    row_ordinal: int
    dependency_sha256: str
    runtime_fields_sha256: str
    payload_sha256: str
    build_status: str
    converter_control_status: str | None
    converter_control_plan_build_attempts: int
    occurrence_count: int
    converter_build_attempts: int
    plan_build_attempts: int
    frontier_work: _FrontierWork
    converter_control_frontier_sha256: str | None
    frontier_sha256: str | None
    sha256: str


class ExactRowExecutionBinding:
    """Opaque authority over one complete desired-free V3 row execution."""

    __slots__ = ()

    def __new__(cls) -> ExactRowExecutionBinding:
        raise TypeError("exact V3 execution bindings require private authority")

    @property
    def row_ordinal(self) -> int:
        return _lookup_binding(self).row_ordinal

    @property
    def status(self) -> str:
        return _lookup_binding(self).build_status

    @property
    def requires_desired(self) -> bool:
        receipt = _lookup_binding(self)
        return receipt.frontier is not None and receipt.build_status == READY

    @property
    def sha256(self) -> str:
        return _lookup_binding(self).sha256


class RowTreatmentOutcome(NamedTuple):
    row_ordinal: int
    build_status: str
    public_status: str
    selection_mode: str | None
    selection_status: str | None
    source_gain0_status: str
    source_gain1_status: str
    static_off_status: str
    static_on_status: str
    source_candidate_event_count: int
    source_selected_event_count: int
    selector_pass_count: int
    source_query_count: int
    source_fitted_glif_step_count: int
    source_compiled_kernel_call_count: int
    static_pass_count: int
    static_off_query_count: int
    static_on_query_count: int
    static_query_count: int
    static_fitted_glif_step_count: int
    converter_control_status: str | None
    converter_control_plan_build_attempts: int
    converter_build_attempts: int
    plan_build_attempts: int
    runtime_desired_reads: int
    exact_proof_count: int
    witness_count: int
    selected_proof_forest_sha256: str | None
    relation_completed_units: int
    relation_attempted_units: int
    relation_limit: int
    physical_predictions: int
    physical_races: int
    dependency_sha256: str
    payload_sha256: str
    geometry_sha256: str
    source_certificate_sha256s: tuple[str, str]
    source_request_sha256s: tuple[str, str]
    source_candidate_event_root_sha256s: tuple[str, str]
    source_selected_event_root_sha256s: tuple[str, str]
    source_work_sha256s: tuple[str, str]
    static_certificate_sha256s: tuple[str, str]
    static_request_sha256s: tuple[str, str]
    static_view_sha256s: tuple[str | None, str | None]
    static_work_sha256: str
    converter_control_frontier_sha256: str | None
    frontier_sha256: str | None
    binding_sha256: str
    frontier_work_sha256: str | None
    sha256: str


_ISSUED_BINDINGS: dict[int, tuple[ExactRowExecutionBinding, _BindingReceipt]] = {}
_ISSUED_OUTCOMES: dict[int, RowTreatmentOutcome] = {}
_SELECTION_STATES: dict[int, list[str]] = {}
_ROW_EXECUTION_STATES: dict[int, str] = {}


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


def _source_static_record(stage: _SourceStaticStage) -> dict[str, object]:
    source_receipts = (stage.source_gain0_receipt, stage.source_gain1_receipt)
    static_receipts = (stage.static_off_receipt, stage.static_on_receipt)
    return {
        "source_certificate_sha256s": (
            stage.source_gain0_certificate.sha256,
            stage.source_gain1_certificate.sha256,
        ),
        "source_statuses": tuple(receipt.status for receipt in source_receipts),
        "source_request_sha256s": tuple(
            receipt.request_sha256 for receipt in source_receipts
        ),
        "source_candidate_event_root_sha256s": tuple(
            receipt.candidate_event_root_sha256 for receipt in source_receipts
        ),
        "source_selected_event_root_sha256s": tuple(
            receipt.selected_event_root_sha256 for receipt in source_receipts
        ),
        "source_work_sha256s": tuple(
            receipt.physical_work_receipt_sha256 for receipt in source_receipts
        ),
        "static_certificate_sha256s": (
            stage.static_off_certificate.sha256,
            stage.static_on_certificate.sha256,
        ),
        "static_statuses": tuple(receipt.status for receipt in static_receipts),
        "static_request_sha256s": tuple(
            receipt.request_sha256 for receipt in static_receipts
        ),
        "static_view_sha256s": tuple(
            None if receipt.view is None else receipt.view.sha256
            for receipt in static_receipts
        ),
        "candidate_event_count": stage.candidate_event_count,
        "selected_event_count": stage.selected_event_count,
        "source_query_count": stage.source_query_count,
        "source_fitted_glif_step_count": stage.source_fitted_glif_step_count,
        "source_compiled_kernel_call_count": (
            stage.source_compiled_kernel_call_count
        ),
        "static_off_query_count": stage.static_off_query_count,
        "static_on_query_count": stage.static_on_query_count,
        "static_query_count": stage.static_query_count,
        "static_fitted_glif_step_count": stage.static_fitted_glif_step_count,
        "static_work_sha256": stage.static_work_sha256,
        "same_gain1_certificate_for_both_static_passes": True,
        "host_selected_count_admission_gate_present": False,
    }


def _binding_record(receipt: _BindingReceipt) -> dict[str, object]:
    return {
        "preflight_sha256": receipt.preflight.sha256,
        "row_pairing_sha256": protocol._lookup_row(receipt.row).pairing_sha256,
        "payload_sha256": receipt.payload_sha256,
        "geometry_sha256": receipt.geometry.sha256,
        "source_static": _source_static_record(receipt.stage),
        "converter_control_frontier_sha256": (
            receipt.converter_control_frontier_sha256
        ),
        "frontier_sha256": receipt.frontier_sha256,
        "row_ordinal": receipt.row_ordinal,
        "dependency_sha256": receipt.dependency_sha256,
        "runtime_fields_sha256": receipt.runtime_fields_sha256,
        "build_status": receipt.build_status,
        "converter_control_status": receipt.converter_control_status,
        "converter_control_plan_build_attempts": (
            receipt.converter_control_plan_build_attempts
        ),
        "occurrence_count": receipt.occurrence_count,
        "converter_build_attempts": receipt.converter_build_attempts,
        "plan_build_attempts": receipt.plan_build_attempts,
        "frontier_work": receipt.frontier_work,
        "desired_present": False,
        "exact_object_binding": True,
    }


def _issue_binding(receipt: _BindingReceipt) -> ExactRowExecutionBinding:
    binding = object.__new__(ExactRowExecutionBinding)
    _ISSUED_BINDINGS[id(binding)] = (binding, receipt)
    _SELECTION_STATES[id(binding)] = []
    return binding


def _validate_stage(stage: _SourceStaticStage, candidate_count: int) -> None:
    gain0_receipt = source_static._authenticate_source_selection(
        stage.source_gain0_certificate
    )
    gain1_receipt = source_static._authenticate_source_selection(
        stage.source_gain1_certificate
    )
    static_off_receipt = occurrence_static._authenticate_certificate(
        stage.static_off_certificate
    )
    static_on_receipt = occurrence_static._authenticate_certificate(
        stage.static_on_certificate
    )
    selected_count = len(gain1_receipt.selected_events)
    expected_request = _expected_static_request_sha256(gain1_receipt)
    complete_per_pass = _static_queries_for_one_complete_pass(selected_count)
    if (
        gain0_receipt is not stage.source_gain0_receipt
        or gain1_receipt is not stage.source_gain1_receipt
        or static_off_receipt is not stage.static_off_receipt
        or static_on_receipt is not stage.static_on_receipt
        or gain0_receipt.status != source_static.SOURCE_ZERO
        or gain1_receipt.status != source_static.SOURCE_SELECTED
        or len(gain0_receipt.candidates) != candidate_count
        or len(gain1_receipt.candidates) != candidate_count
        or gain0_receipt.request_sha256 != gain1_receipt.request_sha256
        or gain0_receipt.candidate_event_root_sha256
        != gain1_receipt.candidate_event_root_sha256
        or len(gain0_receipt.selected_events) != 0
        or not 1 <= selected_count <= candidate_count
        or static_off_receipt.request_sha256 != expected_request
        or static_on_receipt.request_sha256 != expected_request
        or static_off_receipt.status != occurrence_static.STOP_ZERO
        or static_off_receipt.view is not None
        or stage.candidate_event_count != candidate_count
        or stage.selected_event_count != selected_count
        or stage.source_query_count != 2 * candidate_count
        or stage.source_fitted_glif_step_count
        != stage.source_query_count * occurrence_static.OUTPUT_STEPS
        or stage.source_compiled_kernel_call_count != 2
        or stage.static_off_query_count
        != _exact_static_query_count(static_off_receipt, selected_count)
        or stage.static_on_query_count
        != _exact_static_query_count(static_on_receipt, selected_count)
        or stage.static_query_count
        != stage.static_off_query_count + stage.static_on_query_count
        or stage.static_off_query_count > complete_per_pass
        or stage.static_on_query_count > complete_per_pass
        or stage.static_fitted_glif_step_count
        != stage.static_query_count * occurrence_static.OUTPUT_STEPS
        or stage.static_work_sha256
        != _digest(
            "static-work",
            {
                "gain1_source_certificate_sha256": (
                    stage.source_gain1_certificate.sha256
                ),
                "selected_event_count": selected_count,
                "static_off_certificate_sha256": stage.static_off_certificate.sha256,
                "static_off_query_count": stage.static_off_query_count,
                "static_on_certificate_sha256": stage.static_on_certificate.sha256,
                "static_on_query_count": stage.static_on_query_count,
                "fitted_glif_step_count": stage.static_fitted_glif_step_count,
                "complete_query_bound_per_pass": complete_per_pass,
            },
        )
    ):
        raise BoundExecutionStop(
            STOP_BOUND_EXECUTION, "source/static stage changed after exact binding"
        )


def _lookup_binding(binding: object) -> _BindingReceipt:
    if type(binding) is not ExactRowExecutionBinding:
        raise BoundExecutionStop(
            protocol.STOP_PROMPT_IO_PAIRING_MISMATCH,
            "V3 execution binding has the wrong type",
        )
    issued = _ISSUED_BINDINGS.get(id(binding))
    if issued is None or issued[0] is not binding:
        raise BoundExecutionStop(
            protocol.STOP_PROMPT_IO_PAIRING_MISMATCH,
            "V3 execution binding is not an issued exact object",
        )
    receipt = issued[1]
    access = protocol._lookup_build_access(receipt.payload)
    row_receipt = protocol._lookup_row(receipt.row)
    if (
        access.preflight is not receipt.preflight
        or access.row is not receipt.row
        or access.build_sha256 != receipt.payload_sha256
        or row_receipt.geometry is not receipt.geometry
        or receipt.geometry
        is not protocol.CANDIDATE_GEOMETRY_RECEIPTS[receipt.row_ordinal]
        or _runtime_fields_sha256(receipt.payload) != receipt.runtime_fields_sha256
    ):
        raise BoundExecutionStop(
            protocol.STOP_PROMPT_IO_PAIRING_MISMATCH,
            "V3 protocol payload or geometry binding changed",
        )
    _validate_stage(receipt.stage, receipt.geometry.total_candidate_count)
    if receipt.converter_control_frontier is not None:
        receipt.converter_control_frontier._authenticate()
        if (
            receipt.converter_control_frontier.sha256
            != receipt.converter_control_frontier_sha256
        ):
            raise BoundExecutionStop(
                STOP_BOUND_EXECUTION, "converter control frontier changed"
            )
    if receipt.frontier is not None:
        receipt.frontier._authenticate()
        if (
            receipt.frontier.sha256 != receipt.frontier_sha256
            or _frontier_work(receipt.frontier.work) != receipt.frontier_work
        ):
            raise BoundExecutionStop(STOP_BOUND_EXECUTION, "packed frontier changed")
    elif receipt.frontier_work != _ZERO_FRONTIER_WORK:
        raise BoundExecutionStop(
            STOP_BOUND_EXECUTION, "frontier work escaped without a frontier"
        )
    if receipt.sha256 != _digest("exact-binding", _binding_record(receipt)):
        raise BoundExecutionStop(STOP_BOUND_EXECUTION, "exact V3 binding changed")
    return receipt


def authenticate_binding(binding: ExactRowExecutionBinding) -> _BindingReceipt:
    """Authenticate and expose the private receipt to the exclusive runner."""

    return _lookup_binding(binding)


def _forge_unissued_static_certificate(
    certificate: occurrence_static.OccurrenceFactorStaticCertificate,
) -> occurrence_static.OccurrenceFactorStaticCertificate:
    occurrence_static._authenticate_certificate(certificate)
    forged = object.__new__(occurrence_static.OccurrenceFactorStaticCertificate)
    forged._receipt = certificate._receipt
    if occurrence_static._ISSUED_CERTIFICATES.get(id(forged)) is not None:
        raise RuntimeError("converter control unexpectedly issued its forged certificate")
    return forged


def bind_row(
    preflight: protocol.FrozenPairingPreflight,
    row: protocol.FrozenPublicTrainJointRow,
    row_ordinal: int,
) -> ExactRowExecutionBinding:
    """Run one complete desired-free row pipeline through its authentic STOP/READY."""

    if type(row_ordinal) is not int or not 0 <= row_ordinal < len(protocol.ROWS):
        raise ValueError("row ordinal is outside the frozen V3 roster")
    if protocol.ROWS[row_ordinal] is not row:
        raise BoundExecutionStop(
            protocol.STOP_PROMPT_IO_PAIRING_MISMATCH,
            "row ordinal substituted a different exact V3 row",
        )
    dependency_sha = authenticate_dependencies()
    payload = protocol.require_build_payload(preflight, row)
    access = protocol._lookup_build_access(payload)
    if access.preflight is not preflight or access.row is not row:
        raise BoundExecutionStop(
            protocol.STOP_PROMPT_IO_PAIRING_MISMATCH,
            "protocol issued a payload outside its exact V3 row preflight",
        )
    geometry = protocol.CANDIDATE_GEOMETRY_RECEIPTS[row_ordinal]
    if protocol._lookup_row(row).geometry is not geometry:
        raise BoundExecutionStop(
            protocol.STOP_PROMPT_IO_PAIRING_MISMATCH,
            "V3 row geometry changed before execution",
        )
    if id(row) in _ROW_EXECUTION_STATES:
        raise BoundExecutionStop(
            STOP_BOUND_EXECUTION, "exact V3 row execution was already attempted"
        )
    # Reserve before the first physical pass: an interrupted or failed row is
    # not silently replayable in the same one-shot process.
    _ROW_EXECUTION_STATES[id(row)] = "running"
    runtime_sha = _runtime_fields_sha256(payload)
    stage = _run_source_static_stage(
        payload.span_records,
        payload.literal_records,
        payload.signature_role_record,
        expected_candidate_count=geometry.total_candidate_count,
    )

    converter_control = None
    frontier = None
    converter_control_status = None
    converter_control_plan_attempts = 0
    converter_build_attempts = 0
    plan_build_attempts = 0
    occurrence_count = 0
    frontier_sha = None
    converter_control_sha = None
    frontier_work = _ZERO_FRONTIER_WORK
    static_on_receipt = stage.static_on_receipt

    if static_on_receipt.status == occurrence_static.READY:
        if static_on_receipt.view is None:
            raise BoundExecutionStop(
                STOP_BOUND_EXECUTION, "static-on READY omitted its authentic view"
            )
        occurrence_count = len(static_on_receipt.view.occurrence_carriers)
        if occurrence_count != stage.selected_event_count:
            raise BoundExecutionStop(
                STOP_BOUND_EXECUTION,
                "static-on occurrence count differs from selected source count",
            )
        forged = _forge_unissued_static_certificate(stage.static_on_certificate)
        converter_control = (
            packed_converter.PackedPublicTrainFitOccurrenceFactorAuthorizedPlanSliceV1.build(
                payload.signature,
                payload.cases,
                payload.literals,
                payload.max_cost,
                forged,
            )
        )
        converter_control._authenticate()
        converter_control_status = converter_control.status
        converter_control_plan_attempts = converter_control.state.plan_build_attempts
        converter_control_sha = converter_control.sha256
        if (
            converter_control_status != STOP_SOURCE_CERTIFICATE
            or converter_control_plan_attempts != 0
            or converter_control.work is not None
        ):
            raise BoundExecutionStop(
                STOP_BOUND_EXECUTION,
                "converter authority control accepted an unissued certificate",
            )
        frontier = (
            packed_converter.PackedPublicTrainFitOccurrenceFactorAuthorizedPlanSliceV1.build(
                payload.signature,
                payload.cases,
                payload.literals,
                payload.max_cost,
                stage.static_on_certificate,
            )
        )
        frontier._authenticate()
        converter_build_attempts = 1
        plan_build_attempts = frontier.state.plan_build_attempts
        frontier_sha = frontier.sha256
        frontier_work = _frontier_work(frontier.work)
        if (
            frontier.state.desired_reads_before_select != 0
            or frontier.state.proof_witnesses_before_select != 0
            or frontier_work.concrete_proof_witnesses != 0
        ):
            raise BoundExecutionStop(
                STOP_BOUND_EXECUTION,
                "desired-free V3 frontier consumed desired or proof evidence",
            )
        build_status = frontier.status
    else:
        # This is a complete scientific stage outcome, not an integrity error.
        build_status = static_on_receipt.status

    if _runtime_fields_sha256(payload) != runtime_sha:
        raise BoundExecutionStop(
            protocol.STOP_PROMPT_IO_PAIRING_MISMATCH,
            "V3 runtime build fields changed during physical construction",
        )
    partial = _BindingReceipt(
        preflight,
        row,
        payload,
        geometry,
        stage,
        converter_control,
        frontier,
        row_ordinal,
        dependency_sha,
        runtime_sha,
        payload.sha256,
        build_status,
        converter_control_status,
        converter_control_plan_attempts,
        occurrence_count,
        converter_build_attempts,
        plan_build_attempts,
        frontier_work,
        converter_control_sha,
        frontier_sha,
        "",
    )
    receipt = partial._replace(
        sha256=_digest("exact-binding", _binding_record(partial))
    )
    binding = _issue_binding(receipt)
    _lookup_binding(binding)
    _ROW_EXECUTION_STATES[id(row)] = "bound"
    return binding


def _issue_outcome(
    receipt: _BindingReceipt,
    *,
    public_status: str,
    selection_mode: str | None,
    selection_status: str | None,
    runtime_desired_reads: int,
    exact_proof_count: int,
    witness_count: int,
    selected_proof_forest_sha256: str | None,
    work: _FrontierWork,
) -> RowTreatmentOutcome:
    stage = receipt.stage
    sources = (stage.source_gain0_receipt, stage.source_gain1_receipt)
    statics = (stage.static_off_receipt, stage.static_on_receipt)
    values = (
        receipt.row_ordinal,
        receipt.build_status,
        public_status,
        selection_mode,
        selection_status,
        sources[0].status,
        sources[1].status,
        statics[0].status,
        statics[1].status,
        stage.candidate_event_count,
        stage.selected_event_count,
        2,
        stage.source_query_count,
        stage.source_fitted_glif_step_count,
        stage.source_compiled_kernel_call_count,
        2,
        stage.static_off_query_count,
        stage.static_on_query_count,
        stage.static_query_count,
        stage.static_fitted_glif_step_count,
        receipt.converter_control_status,
        receipt.converter_control_plan_build_attempts,
        receipt.converter_build_attempts,
        receipt.plan_build_attempts,
        runtime_desired_reads,
        exact_proof_count,
        witness_count,
        selected_proof_forest_sha256,
        work.relation_completed_units,
        work.relation_attempted_units,
        work.relation_limit,
        work.physical_predictions,
        work.physical_races,
        receipt.dependency_sha256,
        receipt.payload_sha256,
        receipt.geometry.sha256,
        (
            stage.source_gain0_certificate.sha256,
            stage.source_gain1_certificate.sha256,
        ),
        tuple(source.request_sha256 for source in sources),
        tuple(source.candidate_event_root_sha256 for source in sources),
        tuple(source.selected_event_root_sha256 for source in sources),
        tuple(source.physical_work_receipt_sha256 for source in sources),
        (stage.static_off_certificate.sha256, stage.static_on_certificate.sha256),
        tuple(static.request_sha256 for static in statics),
        tuple(
            None if static.view is None else static.view.sha256 for static in statics
        ),
        stage.static_work_sha256,
        receipt.converter_control_frontier_sha256,
        receipt.frontier_sha256,
        receipt.sha256,
        work.sha256,
    )
    outcome = RowTreatmentOutcome(*values, _digest("row-outcome", values))
    _ISSUED_OUTCOMES[id(outcome)] = outcome
    return outcome


def finish_bound(binding: ExactRowExecutionBinding) -> RowTreatmentOutcome:
    """Publish one complete STOP outcome without opening desired values."""

    receipt = _lookup_binding(binding)
    if binding.requires_desired:
        raise BoundExecutionStop(
            STOP_BOUND_EXECUTION,
            "READY V3 binding requires runner-owned wrong/normal selection",
        )
    if receipt.frontier_work.concrete_proof_witnesses != 0:
        raise BoundExecutionStop(
            STOP_BOUND_EXECUTION, "desired-free V3 STOP retained proof witnesses"
        )
    return _issue_outcome(
        receipt,
        public_status=receipt.build_status,
        selection_mode=None,
        selection_status=None,
        runtime_desired_reads=0,
        exact_proof_count=0,
        witness_count=0,
        selected_proof_forest_sha256=None,
        work=receipt.frontier_work,
    )


def select_bound(
    binding: ExactRowExecutionBinding,
    desired: Sequence[object],
    *,
    mode: str,
) -> RowTreatmentOutcome:
    """Select only after an exact desired-free READY binding."""

    receipt = _lookup_binding(binding)
    if mode not in SELECTION_ORDER:
        raise ValueError(f"unknown V3 selection mode: {mode}")
    if receipt.frontier is None or not binding.requires_desired:
        raise BoundExecutionStop(
            STOP_BOUND_EXECUTION, "V3 binding is not eligible for desired selection"
        )
    completed = _SELECTION_STATES[id(binding)]
    expected_index = len(completed)
    if expected_index >= len(SELECTION_ORDER) or mode != SELECTION_ORDER[expected_index]:
        raise BoundExecutionStop(
            STOP_BOUND_EXECUTION, "V3 wrong-desired/normal order changed"
        )

    frozen_desired = tuple(desired)
    selection = receipt.frontier.select(frozen_desired)
    _lookup_binding(binding)
    if selection.build_state_sha256 != receipt.frontier_sha256:
        raise BoundExecutionStop(
            protocol.STOP_PROMPT_IO_PAIRING_MISMATCH,
            "selection is not bound to the exact V3 frontier",
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
                STOP_BOUND_EXECUTION,
                "authorized mismatch exposed forbidden proof evidence",
            )
        public_status = protocol.STOP_PROMPT_IO_PAIRING_MISMATCH
    witnesses = tuple(selection.witnesses)
    work = _frontier_work(selection.work)
    if work.concrete_proof_witnesses != len(witnesses):
        raise BoundExecutionStop(
            STOP_BOUND_EXECUTION, "selection work differs from published witnesses"
        )
    completed.append(mode)
    return _issue_outcome(
        receipt,
        public_status=public_status,
        selection_mode=mode,
        selection_status=selection_status,
        runtime_desired_reads=len(frozen_desired),
        exact_proof_count=selection.exact_proof_count,
        witness_count=len(witnesses),
        selected_proof_forest_sha256=selection.selected_proof_forest_sha256,
        work=work,
    )


def authenticate_outcome(outcome: RowTreatmentOutcome) -> RowTreatmentOutcome:
    """Authenticate one exact immutable outcome issued by this engine."""

    if (
        type(outcome) is not RowTreatmentOutcome
        or _ISSUED_OUTCOMES.get(id(outcome)) is not outcome
    ):
        raise TypeError("V3 row outcome is not an issued exact object")
    if outcome.sha256 != _digest("row-outcome", outcome[:-1]):
        raise RuntimeError("V3 row outcome receipt changed")
    return outcome


CLAIM_BOUNDARY = (
    "fresh public-TRAIN same-FIT V3 exact source/static/converter execution "
    "boundary only: two source passes, same-certificate static off/on, authentic "
    "STOP retention, dynamic-n work accounting, and desired-free packed binding; "
    "no V3 row has been executed by this source artifact itself, and no result, "
    "spend, unseen prompt/family transfer, heldout, DEV, M0, C1, G0, protected, "
    "SWE-bench, or coding-capability evidence is claimed"
)


__all__ = [
    "CLAIM_BOUNDARY",
    "MODE_NORMAL",
    "MODE_WRONG_DESIRED",
    "SCHEMA",
    "SELECTION_ORDER",
    "STOP_BOUND_EXECUTION",
    "BoundExecutionStop",
    "ExactRowExecutionBinding",
    "RowTreatmentOutcome",
    "authenticate_binding",
    "authenticate_dependencies",
    "authenticate_outcome",
    "bind_row",
    "finish_bound",
    "select_bound",
]
