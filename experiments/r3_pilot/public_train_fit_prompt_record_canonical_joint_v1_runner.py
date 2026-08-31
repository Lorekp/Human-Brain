"""Prepared one-shot runner for the two-row public-TRAIN prompt/I-O joint.

The public interface is the no-argument :func:`run_one_shot`.  It creates an
exclusive durable spend marker immediately before the first real build, runs
the frozen controls and then normal row A followed by row B, and publishes one
exclusive immutable result.

Every build is owned by this module.  A protocol payload is opened internally,
its exact fields are passed directly to the banked packed frontier, and the
resulting exact frontier is registered with that exact row and payload.  Only
an authenticated binding may open the row's private desired tuple.  Matching
shape, counts, hashes, or a caller-supplied frontier are never sufficient.

Importing this module spends nothing and creates no files.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import NamedTuple

from experiments.r3_pilot import (
    public_train_fit_occurrence_factor_static_checkpoint_capsule_v1 as occurrence_static,
)
from experiments.r3_pilot import (
    public_train_fit_prompt_record_canonical_joint_v1_protocol as protocol,
)
from experiments.r3_pilot import (
    public_train_fit_prompt_record_source_event_producer_v1 as prompt_producer,
)
from experiments.r3_pilot import (
    typed_semantic_packed_public_train_fit_occurrence_factor_authorized_plan_slice_v1,
    typed_semantic_packed_public_train_fit_prompt_record_authorized_plan_frontier_v1,
)

packed_converter = (
    typed_semantic_packed_public_train_fit_occurrence_factor_authorized_plan_slice_v1
)
packed_frontier = (
    typed_semantic_packed_public_train_fit_prompt_record_authorized_plan_frontier_v1
)

SCHEMA = "PUBLIC_TRAIN_FIT_PROMPT_RECORD_CANONICAL_JOINT_V1_RUNNER"
PRESPEND_READY = "PUBLIC_TRAIN_FIT_PROMPT_RECORD_CANONICAL_JOINT_V1_PRESPEND_READY"
STOP_PRESPEND = "STOP_PUBLIC_TRAIN_FIT_PROMPT_RECORD_CANONICAL_JOINT_V1_PRESPEND"
STOP_WORK_CAP = "STOP_PUBLIC_TRAIN_FIT_PROMPT_RECORD_CANONICAL_JOINT_V1_WORK_CAP"
STOP_ALREADY_SPENT = (
    "STOP_PUBLIC_TRAIN_FIT_PROMPT_RECORD_CANONICAL_JOINT_V1_ALREADY_SPENT"
)

PROTOCOL_BANK_COMMIT = "c9ae0fdad7534a5b9ed44e6cab8d057afcd1f7f0"
PROTOCOL_SOURCE_SHA256 = (
    "a149b3468a980517c30d7322a4247b9f7506306fc80f2ced4d0ff9ab479a97a0"
)
PROTOCOL_ROOT_SHA256 = (
    "524947ef3a29dbb4a7292c4e386d69090d9027c72fccb9559409aff1cf9cc6d0"
)
PROTOCOL_OUTCOME_ROOT_SHA256 = (
    "3ed755cebe6bfc76c865733ee83a2edb5f712fa9885238e24e0437c904ce580c"
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path(__file__).with_suffix(".result.json")
SPENT_PATH = Path(__file__).with_suffix(".spent.json")
RUN_COMMAND = (
    r".\.venv\Scripts\python.exe -m "
    "experiments.r3_pilot."
    "public_train_fit_prompt_record_canonical_joint_v1_runner"
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

_NAMESPACE = b"public-train-fit-prompt-record-canonical-joint-v1-runner\0"
_DEPENDENCY_SHA256S = {
    "experiments/r3_pilot/public_train_fit_prompt_record_canonical_joint_v1_protocol.py": (
        PROTOCOL_SOURCE_SHA256
    ),
    (
        "experiments/r3_pilot/typed_semantic_packed_public_train_fit_"
        "prompt_record_authorized_plan_frontier_v1.py"
    ): (
        "e16e589a8fabf4f63ec8540ff8670f87bbc5b7f041277c48df67003da99ad5f4"
    ),
    "experiments/r3_pilot/public_train_fit_prompt_record_source_event_producer_v1.py": (
        "5e195bccf8c8ebbe59bfc6d89f6f27b9fc15ee6fe8a6a62cbe96f57650d04784"
    ),
    "experiments/r3_pilot/public_train_fit_occurrence_factor_static_checkpoint_capsule_v1.py": (
        "0af68609e46afd65e8dc50c656f7ad22919a08329045d3068f8a457813870c89"
    ),
    (
        "experiments/r3_pilot/typed_semantic_packed_public_train_fit_"
        "occurrence_factor_authorized_plan_slice_v1.py"
    ): (
        "daa8a119caec7c2700c46fadb569676fed7742413899856de6d6e2d12bf5a90c"
    ),
}


def _plain(value: object) -> object:
    if isinstance(value, Path):
        return value.resolve().relative_to(REPO_ROOT).as_posix()
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


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _display_path(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


PRESPEND_CONTRACT = {
    "schema": SCHEMA,
    "protocol_bank_commit": PROTOCOL_BANK_COMMIT,
    "protocol_source_sha256": PROTOCOL_SOURCE_SHA256,
    "protocol_root_sha256": PROTOCOL_ROOT_SHA256,
    "protocol_outcome_root_sha256": PROTOCOL_OUTCOME_ROOT_SHA256,
    "treatment_order": TREATMENT_ORDER,
    "row_order": (0, 1),
    "normal_order": ("row-A", "row-B"),
    "producer_query_cap": protocol.PRODUCER_QUERY_CAP,
    "static_query_cap": protocol.STATIC_QUERY_CAP,
    "fitted_glif_step_cap": protocol.FITTED_GLIF_STEP_CAP,
    "per_plan_work_caps": protocol.PLAN_WORK_CAPS,
    "aggregate_plan_work_caps": protocol.AGGREGATE_PLAN_WORK_CAPS,
    "result_path": _display_path(RESULT_PATH),
    "spent_path": _display_path(SPENT_PATH),
    "retry_allowed": False,
    "search_allowed": False,
    "cap_raise_allowed": False,
    "row_substitution_allowed": False,
    "shape_only_binding_allowed": False,
}
PRESPEND_CONTRACT_SHA256 = _digest("prespend-contract", PRESPEND_CONTRACT)


class RunnerStop(RuntimeError):
    """Fail-closed runner error with one public status."""

    def __init__(self, status: str, reason: str) -> None:
        self.status = status
        super().__init__(reason)


class _BuildProduct(NamedTuple):
    payload: protocol.FrozenJointBuildPayload
    frontier: object
    runtime_fields_sha256: str


class _Backend(NamedTuple):
    name: str
    real: bool
    build: Callable[[protocol.FrozenJointBuildPayload, str], _BuildProduct]
    authenticate: Callable[[object], None]
    select: Callable[[object, tuple[int, ...]], object]


class _BindingReceipt(NamedTuple):
    preflight: protocol.FrozenPairingPreflight
    row: protocol.FrozenPublicTrainJointRow
    payload: protocol.FrozenJointBuildPayload
    frontier: object
    backend_name: str
    mode: str
    payload_sha256: str
    runtime_fields_sha256: str
    frontier_sha256: str
    sha256: str


class _ExactPayloadFrontierBinding:
    """Opaque exact row + payload + executed-frontier authority."""

    __slots__ = ()

    def __new__(cls) -> _ExactPayloadFrontierBinding:
        raise TypeError("exact build bindings require private runner authority")


class _PreparedReceipt(NamedTuple):
    exact_preflight: protocol.FrozenPairingPreflight
    forgery_preflight: protocol.FrozenPairingPreflight
    dependency_sha256: str
    sha256: str


class _PreparedRun:
    """Opaque authenticated pre-spend state; no electrical work has run."""

    __slots__ = ()

    def __new__(cls) -> _PreparedRun:
        raise TypeError("prepared runs require private runner authority")

    @property
    def sha256(self) -> str:
        return _lookup_prepared(self).sha256


class _AuthorityReceipt(NamedTuple):
    prepared: _PreparedRun
    real_spend: bool
    backend_name: str
    marker_sha256: str | None
    sha256: str


class _ExecutionAuthority:
    """Opaque authority proving a durable spend or a stub-only test run."""

    __slots__ = ()

    def __new__(cls) -> _ExecutionAuthority:
        raise TypeError("execution authorities require private runner authority")


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
    frontier_sha256: str
    binding_sha256: str
    work_sha256: str | None
    sha256: str


class WorkLedger(NamedTuple):
    producer_query_count: int
    static_query_count: int
    fitted_glif_step_count: int
    plan_build_count: int
    relation_work_units: int
    physical_predictions: int
    physical_races: int
    sha256: str


class RunnerExecution(NamedTuple):
    verdict: str
    stop_reason: str | None
    prepared_sha256: str
    treatment_outcomes: tuple[TreatmentOutcome, ...]
    work: WorkLedger
    proof_materialization_count: int
    result_artifact_count_before_return: int
    sha256: str


_ISSUED_BINDINGS: dict[
    int, tuple[_ExactPayloadFrontierBinding, _BindingReceipt]
] = {}
_ISSUED_PREPARED: dict[int, tuple[_PreparedRun, _PreparedReceipt]] = {}
_ISSUED_AUTHORITIES: dict[
    int, tuple[_ExecutionAuthority, _AuthorityReceipt]
] = {}
_INTERVENTION_ACTIVE = False


def _issue_prepared(receipt: _PreparedReceipt) -> _PreparedRun:
    prepared = object.__new__(_PreparedRun)
    _ISSUED_PREPARED[id(prepared)] = (prepared, receipt)
    return prepared


def _lookup_prepared(prepared: object) -> _PreparedReceipt:
    if type(prepared) is not _PreparedRun:
        raise RunnerStop(STOP_PRESPEND, "prepared run has the wrong type")
    issued = _ISSUED_PREPARED.get(id(prepared))
    if issued is None or issued[0] is not prepared:
        raise RunnerStop(STOP_PRESPEND, "prepared run is not an issued exact object")
    return issued[1]


def _issue_authority(receipt: _AuthorityReceipt) -> _ExecutionAuthority:
    authority = object.__new__(_ExecutionAuthority)
    _ISSUED_AUTHORITIES[id(authority)] = (authority, receipt)
    return authority


def _lookup_authority(authority: object) -> _AuthorityReceipt:
    if type(authority) is not _ExecutionAuthority:
        raise RunnerStop(STOP_PRESPEND, "execution authority has the wrong type")
    issued = _ISSUED_AUTHORITIES.get(id(authority))
    if issued is None or issued[0] is not authority:
        raise RunnerStop(STOP_PRESPEND, "execution authority is not an issued object")
    return issued[1]


def _lookup_binding(
    binding: object,
    backend: _Backend,
) -> _BindingReceipt:
    if type(binding) is not _ExactPayloadFrontierBinding:
        raise RunnerStop(
            protocol.STOP_PROMPT_IO_PAIRING_MISMATCH,
            "payload-frontier binding has the wrong type",
        )
    issued = _ISSUED_BINDINGS.get(id(binding))
    if issued is None or issued[0] is not binding:
        raise RunnerStop(
            protocol.STOP_PROMPT_IO_PAIRING_MISMATCH,
            "payload-frontier binding is not an issued exact object",
        )
    receipt = issued[1]
    if receipt.backend_name != backend.name:
        raise RunnerStop(
            protocol.STOP_PROMPT_IO_PAIRING_MISMATCH,
            "payload-frontier binding changed backend authority",
        )
    access = protocol._lookup_build_access(receipt.payload)
    if (
        access.preflight is not receipt.preflight
        or access.row is not receipt.row
        or access.build_sha256 != receipt.payload_sha256
    ):
        raise RunnerStop(
            protocol.STOP_PROMPT_IO_PAIRING_MISMATCH,
            "protocol payload binding changed",
        )
    backend.authenticate(receipt.frontier)
    if _frontier_sha(receipt.frontier) != receipt.frontier_sha256:
        raise RunnerStop(
            protocol.STOP_PROMPT_IO_PAIRING_MISMATCH,
            "executed frontier changed after exact binding",
        )
    return receipt


def _runtime_field_record(
    span_records: Sequence[Mapping[str, object]],
    literal_records: Sequence[Mapping[str, object]],
    signature_role_record: Mapping[str, object],
    signature: object,
    cases: Sequence[tuple[object, ...]],
    literals: Sequence[int],
    max_cost: int,
) -> dict[str, object]:
    return {
        "span_records": span_records,
        "literal_records": literal_records,
        "signature_role_record": signature_role_record,
        "signature": signature,
        "cases": cases,
        "literals": literals,
        "max_cost": max_cost,
        "desired_present": False,
    }


def _payload_field_sha256(
    payload: protocol.FrozenJointBuildPayload,
) -> str:
    return _digest(
        "runtime-build-fields",
        _runtime_field_record(
            payload.span_records,
            payload.literal_records,
            payload.signature_role_record,
            payload.signature,
            payload.cases,
            payload.literals,
            payload.max_cost,
        ),
    )


@contextmanager
def _real_intervention(mode: str):
    global _INTERVENTION_ACTIVE
    if _INTERVENTION_ACTIVE:
        raise RuntimeError("one-shot interventions cannot overlap")
    _INTERVENTION_ACTIVE = True
    def restore() -> None:
        pass

    try:
        if mode == MODE_PRODUCER_OFF:
            original = prompt_producer._is_selected

            def cell_off(event: object, *, cell_gain: float = 1.0) -> bool:
                del cell_gain
                return original(event, cell_gain=0.0)

            prompt_producer._is_selected = cell_off

            def restore() -> None:
                prompt_producer._is_selected = original

        elif mode == MODE_STATIC_OFF:
            original = occurrence_static._decode

            def cell_off(query: object, *, cell_gain: float = 1.0) -> object:
                del cell_gain
                return original(query, cell_gain=0.0)

            occurrence_static._decode = cell_off

            def restore() -> None:
                occurrence_static._decode = original

        elif mode == MODE_CONVERTER_OFF:
            converter_type = (
                packed_converter.PackedPublicTrainFitOccurrenceFactorAuthorizedPlanSliceV1
            )
            original_descriptor = converter_type.__dict__["build"]
            original_build = converter_type.build

            def converter_off(
                cls: type,
                signature: object,
                cases: object,
                literals: object,
                max_cost: int,
                source_certificate: object,
            ) -> object:
                if type(source_certificate) is not (
                    occurrence_static.OccurrenceFactorStaticCertificate
                ):
                    raise TypeError("converter-off received the wrong certificate type")
                unissued = object.__new__(type(source_certificate))
                unissued._receipt = source_certificate._receipt
                stopped = original_build(
                    signature,
                    cases,
                    literals,
                    max_cost,
                    unissued,
                )
                if (
                    type(stopped) is not cls
                    or stopped.status != packed_converter.STOP_SOURCE_CERTIFICATE
                ):
                    raise RuntimeError(
                        "banked converter did not reject the unissued certificate"
                    )
                stopped._authenticate()
                return stopped

            converter_type.build = classmethod(converter_off)

            def restore() -> None:
                converter_type.build = original_descriptor

        elif mode not in (MODE_WRONG_DESIRED, MODE_NORMAL):
            raise ValueError(f"unknown frozen one-shot mode: {mode}")
        yield
    finally:
        restore()
        _INTERVENTION_ACTIVE = False


def _real_build(
    payload: protocol.FrozenJointBuildPayload,
    mode: str,
) -> _BuildProduct:
    span_records = payload.span_records
    literal_records = payload.literal_records
    signature_role_record = payload.signature_role_record
    signature = payload.signature
    cases = payload.cases
    literals = payload.literals
    max_cost = payload.max_cost
    fields = _runtime_field_record(
        span_records,
        literal_records,
        signature_role_record,
        signature,
        cases,
        literals,
        max_cost,
    )
    fields_sha = _digest("runtime-build-fields", fields)
    with _real_intervention(mode):
        frontier = (
            packed_frontier.PackedPublicTrainFitPromptRecordAuthorizedPlanFrontierV1.build(
                span_records,
                literal_records,
                signature_role_record,
                signature,
                cases,
                literals,
                max_cost,
            )
        )
    if _digest("runtime-build-fields", fields) != fields_sha:
        raise RuntimeError("exact runtime build fields changed during construction")
    return _BuildProduct(payload, frontier, fields_sha)


def _real_authenticate(frontier: object) -> None:
    if type(frontier) is not (
        packed_frontier.PackedPublicTrainFitPromptRecordAuthorizedPlanFrontierV1
    ):
        raise TypeError("real runner received a non-banked frontier")
    frontier._authenticate()


def _real_select(frontier: object, desired: tuple[int, ...]) -> object:
    _real_authenticate(frontier)
    return frontier.select(desired)


_REAL_BACKEND = _Backend(
    "banked-prompt-record-frontier-v1",
    True,
    _real_build,
    _real_authenticate,
    _real_select,
)


def _frontier_sha(frontier: object) -> str:
    value = getattr(frontier, "sha256", None)
    if type(value) is not str or len(value) != 64:
        raise TypeError("frontier has no exact SHA-256 receipt")
    return value


def _build_bound(
    preflight: protocol.FrozenPairingPreflight,
    row: protocol.FrozenPublicTrainJointRow,
    mode: str,
    backend: _Backend,
) -> _ExactPayloadFrontierBinding:
    payload = protocol.require_build_payload(preflight, row)
    access = protocol._lookup_build_access(payload)
    if access.preflight is not preflight or access.row is not row:
        raise RunnerStop(
            protocol.STOP_PROMPT_IO_PAIRING_MISMATCH,
            "protocol issued a payload outside the exact row preflight",
        )
    product = backend.build(payload, mode)
    if product.payload is not payload:
        raise RunnerStop(
            protocol.STOP_PROMPT_IO_PAIRING_MISMATCH,
            "builder substituted a different protocol payload",
        )
    expected_fields_sha = _payload_field_sha256(payload)
    if product.runtime_fields_sha256 != expected_fields_sha:
        raise RunnerStop(
            protocol.STOP_PROMPT_IO_PAIRING_MISMATCH,
            "executed build fields differ from the exact protocol payload",
        )
    backend.authenticate(product.frontier)
    frontier_sha = _frontier_sha(product.frontier)
    payload_sha = payload.sha256
    record = {
        "preflight_sha256": preflight.sha256,
        "payload_sha256": payload_sha,
        "runtime_fields_sha256": product.runtime_fields_sha256,
        "frontier_sha256": frontier_sha,
        "backend_name": backend.name,
        "mode": mode,
        "exact_object_binding": True,
    }
    receipt = _BindingReceipt(
        preflight,
        row,
        payload,
        product.frontier,
        backend.name,
        mode,
        payload_sha,
        product.runtime_fields_sha256,
        frontier_sha,
        _digest("exact-payload-frontier-binding", record),
    )
    binding = object.__new__(_ExactPayloadFrontierBinding)
    _ISSUED_BINDINGS[id(binding)] = (binding, receipt)
    return binding


def _private_desired(
    binding: _ExactPayloadFrontierBinding,
    backend: _Backend,
    *,
    wrong: bool,
) -> tuple[int, ...]:
    receipt = _lookup_binding(binding, backend)
    desired = tuple(protocol._lookup_row(receipt.row).selection.desired)
    if not wrong:
        return desired
    shifted = desired[-1:] + desired[:-1]
    if shifted == desired:
        shifted = tuple(value + index + 1 for index, value in enumerate(desired))
    if shifted == desired:
        raise RuntimeError("frozen desired control failed to change the target")
    return shifted


def _work_values(work: object | None) -> tuple[int, int, int, int, str | None]:
    if work is None:
        return (0, 0, 0, 0, None)
    relation = getattr(work, "relation", None)
    completed = getattr(relation, "completed_units", None)
    predictions = getattr(work, "physical_predictions", None)
    races = getattr(work, "physical_races", None)
    proofs = getattr(work, "concrete_proof_witnesses", None)
    work_sha = getattr(work, "sha256", None)
    if not all(type(value) is int for value in (completed, predictions, races, proofs)):
        raise TypeError("frontier work audit is incomplete")
    if type(work_sha) is not str or len(work_sha) != 64:
        raise TypeError("frontier work audit has no exact SHA-256 receipt")
    return completed, predictions, races, proofs, work_sha


def _static_queries(mode: str, occurrence_count: int) -> int:
    complete = occurrence_count * occurrence_count + 18 * occurrence_count
    if mode == MODE_PRODUCER_OFF:
        return 0
    if mode == MODE_STATIC_OFF:
        return occurrence_count
    return complete * 2


def _selection_public_status(selection: object) -> str:
    status = getattr(selection, "status", None)
    if status == packed_converter.plan_slice.STOP_AUTHORIZED_ECLASS_MISMATCH:
        if (
            getattr(selection, "exact_proof_count", None) != 0
            or tuple(getattr(selection, "witnesses", ()))
            or getattr(selection, "selected_proof_forest_sha256", None) is not None
        ):
            raise RunnerStop(
                protocol.STOP_PROMPT_IO_PAIRING_MISMATCH,
                "authorized e-class mismatch exposed proof evidence",
            )
        return protocol.STOP_PROMPT_IO_PAIRING_MISMATCH
    if type(status) is not str:
        raise TypeError("selection has no public status")
    return status


def _treatment(
    preflight: protocol.FrozenPairingPreflight,
    row: protocol.FrozenPublicTrainJointRow,
    row_ordinal: int,
    mode: str,
    backend: _Backend,
) -> TreatmentOutcome:
    binding = _build_bound(preflight, row, mode, backend)
    binding_receipt = _lookup_binding(binding, backend)
    frontier = binding_receipt.frontier
    state = getattr(frontier, "state", None)
    build_status = getattr(frontier, "status", None)
    if type(build_status) is not str or state is None:
        raise TypeError("frontier build state is incomplete")

    selection = None
    desired_reads = 0
    if mode in (MODE_WRONG_DESIRED, MODE_NORMAL) and build_status == packed_frontier.READY:
        desired = _private_desired(
            binding,
            backend,
            wrong=mode == MODE_WRONG_DESIRED,
        )
        desired_reads = len(desired)
        selection = backend.select(frontier, desired)
        _lookup_binding(binding, backend)
        if getattr(selection, "build_state_sha256", None) != binding_receipt.frontier_sha256:
            raise RunnerStop(
                protocol.STOP_PROMPT_IO_PAIRING_MISMATCH,
                "selection is not bound to the exact executed frontier",
            )

    selection_status = None if selection is None else getattr(selection, "status", None)
    public_status = build_status if selection is None else _selection_public_status(selection)
    proof_count = 0 if selection is None else getattr(selection, "exact_proof_count", None)
    witnesses = () if selection is None else tuple(getattr(selection, "witnesses", ()))
    forest_sha = (
        None
        if selection is None
        else getattr(selection, "selected_proof_forest_sha256", None)
    )
    if type(proof_count) is not int:
        raise TypeError("selection exact proof count is not an integer")
    work = (
        getattr(frontier, "work", None)
        if selection is None
        else getattr(selection, "work", None)
    )
    relation, predictions, races, work_proofs, work_sha = _work_values(work)
    if work_proofs != len(witnesses):
        raise RunnerStop(STOP_WORK_CAP, "work proof count differs from published witnesses")

    expected_occurrences = protocol._lookup_row(row).occurrence_count
    static_queries = _static_queries(mode, expected_occurrences)
    payload = {
        "mode": mode,
        "row_ordinal": row_ordinal,
        "build_status": build_status,
        "source_event_status": getattr(state, "source_event_status", None),
        "occurrence_status": getattr(state, "occurrence_status", None),
        "public_status": public_status,
        "selection_status": selection_status,
        "source_candidate_event_count": getattr(
            state, "source_candidate_event_count", None
        ),
        "source_event_count": getattr(state, "source_event_count", None),
        "occurrence_count": getattr(state, "occurrence_count", None),
        "producer_build_attempts": getattr(state, "producer_build_attempts", None),
        "occurrence_inference_attempts": getattr(
            state, "occurrence_inference_attempts", None
        ),
        "plan_build_attempts": getattr(state, "plan_build_attempts", None),
        "runtime_desired_reads": desired_reads,
        "exact_proof_count": proof_count,
        "witness_count": len(witnesses),
        "selected_proof_forest_sha256": forest_sha,
        "relation_work_units": relation,
        "physical_predictions": predictions,
        "physical_races": races,
        "static_query_count": static_queries,
        "payload_sha256": binding_receipt.payload_sha256,
        "frontier_sha256": binding_receipt.frontier_sha256,
        "binding_sha256": binding_receipt.sha256,
        "work_sha256": work_sha,
    }
    integer_fields = (
        "source_candidate_event_count",
        "source_event_count",
        "occurrence_count",
        "producer_build_attempts",
        "occurrence_inference_attempts",
        "plan_build_attempts",
    )
    if any(type(payload[field]) is not int for field in integer_fields):
        raise TypeError("frontier state counters are incomplete")
    outcome = TreatmentOutcome(
        *(payload[field] for field in TreatmentOutcome._fields if field != "sha256"),
        "",
    )
    return outcome._replace(sha256=_digest("treatment", outcome[:-1]))


def _expected(outcome: TreatmentOutcome, expected_occurrences: int) -> str | None:
    if outcome.source_candidate_event_count != protocol.PRODUCER_CANDIDATES_PER_ACTIVATION:
        return "producer candidate count changed"
    if outcome.producer_build_attempts != 1:
        return "producer build count changed"
    if mode := outcome.mode:
        if mode == MODE_PRODUCER_OFF:
            valid = (
                outcome.build_status == packed_frontier.STOP_SOURCE_EVENT_CERTIFICATE
                and outcome.source_event_status == prompt_producer.STOP_ZERO
                and outcome.source_event_count == 0
                and outcome.occurrence_inference_attempts == 0
                and outcome.plan_build_attempts == 0
                and outcome.runtime_desired_reads == 0
                and outcome.exact_proof_count == 0
            )
        elif mode == MODE_STATIC_OFF:
            valid = (
                outcome.build_status == packed_frontier.STOP_SOURCE_EVENT_CERTIFICATE
                and outcome.source_event_status == prompt_producer.STOP_UNSUPPORTED
                and outcome.source_event_count == 0
                and outcome.occurrence_inference_attempts == 0
                and outcome.plan_build_attempts == 0
                and outcome.runtime_desired_reads == 0
                and outcome.exact_proof_count == 0
            )
        elif mode == MODE_CONVERTER_OFF:
            valid = (
                outcome.build_status == packed_converter.STOP_SOURCE_CERTIFICATE
                and outcome.source_event_status == prompt_producer.READY
                and outcome.occurrence_status == occurrence_static.READY
                and outcome.source_event_count == expected_occurrences
                and outcome.occurrence_count == expected_occurrences
                and outcome.occurrence_inference_attempts == 1
                and outcome.plan_build_attempts == 0
                and outcome.runtime_desired_reads == 0
                and outcome.exact_proof_count == 0
            )
        elif mode == MODE_WRONG_DESIRED:
            valid = (
                outcome.build_status == packed_frontier.READY
                and outcome.source_event_status == prompt_producer.READY
                and outcome.occurrence_status == occurrence_static.READY
                and outcome.source_event_count == expected_occurrences
                and outcome.occurrence_count == expected_occurrences
                and outcome.occurrence_inference_attempts == 1
                and outcome.plan_build_attempts == 1
                and outcome.runtime_desired_reads == protocol.VISIBLE_CASE_COUNT
                and outcome.public_status
                in (
                    packed_converter.plan_slice.STOP_NO_EXACT,
                    protocol.STOP_PROMPT_IO_PAIRING_MISMATCH,
                )
                and outcome.exact_proof_count == 0
                and outcome.witness_count == 0
                and outcome.selected_proof_forest_sha256 is None
            )
        elif mode == MODE_NORMAL:
            valid = (
                outcome.build_status == packed_frontier.READY
                and outcome.source_event_status == prompt_producer.READY
                and outcome.occurrence_status == occurrence_static.READY
                and outcome.source_event_count == expected_occurrences
                and outcome.occurrence_count == expected_occurrences
                and outcome.occurrence_inference_attempts == 1
                and outcome.plan_build_attempts == 1
                and outcome.runtime_desired_reads == protocol.VISIBLE_CASE_COUNT
                and outcome.public_status == packed_frontier.SELECTED
                and outcome.exact_proof_count == 1
                and outcome.witness_count >= 1
                and outcome.selected_proof_forest_sha256 is not None
            )
        else:
            return f"unknown treatment mode {mode}"
        if not valid:
            return f"{mode} row {outcome.row_ordinal} violated its frozen outcome"
    return None


def _ledger(outcomes: Sequence[TreatmentOutcome]) -> WorkLedger:
    producer = sum(item.source_candidate_event_count for item in outcomes)
    static = sum(item.static_query_count for item in outcomes)
    plan_builds = sum(item.plan_build_attempts for item in outcomes)
    relation = sum(item.relation_work_units for item in outcomes)
    predictions = sum(item.physical_predictions for item in outcomes)
    races = sum(item.physical_races for item in outcomes)
    fitted = (producer + static) * occurrence_static.OUTPUT_STEPS
    values = (producer, static, fitted, plan_builds, relation, predictions, races)
    return WorkLedger(*values, _digest("work-ledger", values))


def _work_stop(outcomes: Sequence[TreatmentOutcome]) -> str | None:
    for outcome in outcomes:
        if (
            outcome.relation_work_units > protocol.PLAN_WORK_CAPS[0]
            or outcome.physical_predictions > protocol.PLAN_WORK_CAPS[1]
            or outcome.physical_races > protocol.PLAN_WORK_CAPS[2]
        ):
            return f"{outcome.mode} row {outcome.row_ordinal} exceeded a plan work cap"
    ledger = _ledger(outcomes)
    if (
        ledger.producer_query_count > protocol.PRODUCER_QUERY_CAP
        or ledger.static_query_count > protocol.STATIC_QUERY_CAP
        or ledger.fitted_glif_step_count > protocol.FITTED_GLIF_STEP_CAP
        or ledger.plan_build_count > protocol.WORK_CAPS.plan_build_count
        or ledger.relation_work_units > protocol.AGGREGATE_PLAN_WORK_CAPS[0]
        or ledger.physical_predictions > protocol.AGGREGATE_PLAN_WORK_CAPS[1]
        or ledger.physical_races > protocol.AGGREGATE_PLAN_WORK_CAPS[2]
    ):
        return "aggregate one-shot work cap exceeded"
    return None


def _execution(
    verdict: str,
    reason: str | None,
    prepared_sha256: str,
    outcomes: Sequence[TreatmentOutcome],
) -> RunnerExecution:
    frozen = tuple(outcomes)
    work = _ledger(frozen)
    proof_count = sum(item.witness_count for item in frozen)
    record = {
        "verdict": verdict,
        "stop_reason": reason,
        "prepared_sha256": prepared_sha256,
        "treatment_sha256s": tuple(item.sha256 for item in frozen),
        "work_sha256": work.sha256,
        "proof_materialization_count": proof_count,
        "result_artifact_count_before_return": 0,
    }
    return RunnerExecution(
        verdict,
        reason,
        prepared_sha256,
        frozen,
        work,
        proof_count,
        0,
        _digest("execution", record),
    )


def _authenticate_dependencies() -> str:
    for relative, expected in _DEPENDENCY_SHA256S.items():
        if _file_sha256(REPO_ROOT / relative) != expected:
            raise RunnerStop(STOP_PRESPEND, f"banked runner dependency changed: {relative}")
    if (
        protocol.PROTOCOL.protocol_sha256 != PROTOCOL_ROOT_SHA256
        or protocol.PROTOCOL.outcome_contract_sha256
        != PROTOCOL_OUTCOME_ROOT_SHA256
        or protocol.OUTCOME_CONTRACT["future_private_exact_build_binding_required"]
        is not True
        or protocol.OUTCOME_CONTRACT["shape_only_frontier_binding_allowed"]
        is not False
    ):
        raise RunnerStop(STOP_PRESPEND, "banked joint protocol contract changed")
    relative = _display_path(
        REPO_ROOT
        / "experiments/r3_pilot/"
        "public_train_fit_prompt_record_canonical_joint_v1_protocol.py"
    )
    committed = subprocess.run(
        ["git", "show", f"{PROTOCOL_BANK_COMMIT}:{relative}"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )
    if committed.returncode != 0 or sha256(committed.stdout).hexdigest() != (
        PROTOCOL_SOURCE_SHA256
    ):
        raise RunnerStop(STOP_PRESPEND, "protocol commit/source binding changed")
    return _digest(
        "dependencies",
        {
            "protocol_bank_commit": PROTOCOL_BANK_COMMIT,
            "files": _DEPENDENCY_SHA256S,
            "protocol_root": PROTOCOL_ROOT_SHA256,
            "outcome_root": PROTOCOL_OUTCOME_ROOT_SHA256,
        },
    )


def _assert_paths_absent() -> None:
    for label, path in (("spent marker", SPENT_PATH), ("result artifact", RESULT_PATH)):
        if path.exists():
            raise RunnerStop(STOP_ALREADY_SPENT, f"{label} already exists: {path}")


def _prepare_pre_spend() -> _PreparedRun:
    """Authenticate the frozen roster and forgery STOP without executing it."""

    _assert_paths_absent()
    dependency_sha = _authenticate_dependencies()
    exact = protocol.preflight_pairing(protocol.ROWS)
    if exact.status != protocol.PASS:
        raise RunnerStop(STOP_PRESPEND, "exact atomic roster preflight stopped")
    forged = object.__new__(protocol.FrozenPublicTrainJointRow)
    forgery = protocol.preflight_pairing((forged, protocol.ROWS[1]))
    if (
        forgery.status != protocol.STOP_ROW_CERTIFICATE
        or forgery.electrical_execution_count != 0
        or forgery.proof_materialization_count != 0
        or forgery.runtime_desired_read_count != 0
    ):
        raise RunnerStop(STOP_PRESPEND, "row-forgery preflight did not stop atomically")
    record = {
        "status": PRESPEND_READY,
        "dependency_sha256": dependency_sha,
        "exact_preflight_sha256": exact.sha256,
        "forgery_preflight_sha256": forgery.sha256,
        "electrical_execution_count": 0,
        "proof_materialization_count": 0,
        "runtime_desired_read_count": 0,
        "result_artifact_count": 0,
    }
    receipt = _PreparedReceipt(
        exact,
        forgery,
        dependency_sha,
        _digest("prepared-run", record),
    )
    return _issue_prepared(receipt)


def _write_json_exclusive(path: Path, payload: Mapping[str, object]) -> str:
    serialized = json.dumps(
        _plain(payload),
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise RunnerStop(STOP_ALREADY_SPENT, f"refusing to overwrite {path}") from error
    return _file_sha256(path)


def _create_spend_authority(prepared: _PreparedRun) -> _ExecutionAuthority:
    receipt = _lookup_prepared(prepared)
    _assert_paths_absent()
    marker = {
        "schema": SCHEMA,
        "state": "PUBLIC_TRAIN_JOINT_ATTEMPT_SPENT",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "protocol_bank_commit": PROTOCOL_BANK_COMMIT,
        "protocol_root_sha256": PROTOCOL_ROOT_SHA256,
        "prepared_sha256": receipt.sha256,
        "runner_source_sha256": _file_sha256(Path(__file__)),
        "result_path": _display_path(RESULT_PATH),
        "no_retry": True,
    }
    marker_sha = _write_json_exclusive(SPENT_PATH, marker)
    if RESULT_PATH.exists():
        raise RunnerStop(
            STOP_ALREADY_SPENT,
            "result appeared while spend marker was created; marker retained",
        )
    authority_record = {
        "prepared_sha256": receipt.sha256,
        "real_spend": True,
        "backend_name": _REAL_BACKEND.name,
        "marker_sha256": marker_sha,
    }
    return _issue_authority(
        _AuthorityReceipt(
            prepared,
            True,
            _REAL_BACKEND.name,
            marker_sha,
            _digest("execution-authority", authority_record),
        )
    )


def _issue_stub_test_authority(
    prepared: _PreparedRun,
    backend: _Backend,
) -> _ExecutionAuthority:
    """Issue a no-file authority only for the focused non-real backend test."""

    prepared_receipt = _lookup_prepared(prepared)
    _assert_paths_absent()
    if backend.real:
        raise RunnerStop(STOP_PRESPEND, "stub authority cannot execute the real backend")
    record = {
        "prepared_sha256": prepared_receipt.sha256,
        "real_spend": False,
        "backend_name": backend.name,
        "marker_sha256": None,
    }
    return _issue_authority(
        _AuthorityReceipt(
            prepared,
            False,
            backend.name,
            None,
            _digest("execution-authority", record),
        )
    )


def _authenticate_execution_authority(
    prepared: _PreparedRun,
    authority: _ExecutionAuthority,
    backend: _Backend,
) -> _AuthorityReceipt:
    _lookup_prepared(prepared)
    receipt = _lookup_authority(authority)
    if receipt.prepared is not prepared or receipt.backend_name != backend.name:
        raise RunnerStop(STOP_PRESPEND, "execution authority binding changed")
    if backend.real:
        if not receipt.real_spend or receipt.marker_sha256 is None:
            raise RunnerStop(STOP_PRESPEND, "real backend lacks durable spend authority")
        if not SPENT_PATH.is_file() or _file_sha256(SPENT_PATH) != receipt.marker_sha256:
            raise RunnerStop(STOP_PRESPEND, "durable spend marker changed during run")
    elif receipt.real_spend:
        raise RunnerStop(STOP_PRESPEND, "stub backend received real spend authority")
    return receipt


def _execute_prepared(
    prepared: _PreparedRun,
    authority: _ExecutionAuthority,
    backend: _Backend,
) -> RunnerExecution:
    """Internal test seam; the real backend requires a durable spend authority."""

    prepared_receipt = _lookup_prepared(prepared)
    _authenticate_execution_authority(prepared, authority, backend)
    outcomes: list[TreatmentOutcome] = []
    normal_stops: list[tuple[str, str]] = []
    for mode in TREATMENT_ORDER:
        for row_ordinal, row in enumerate(protocol.ROWS):
            _authenticate_execution_authority(prepared, authority, backend)
            outcome = _treatment(
                prepared_receipt.exact_preflight,
                row,
                row_ordinal,
                mode,
                backend,
            )
            outcomes.append(outcome)
            expected_occurrences = protocol._lookup_row(row).occurrence_count
            expected_stop = _expected(outcome, expected_occurrences)
            work_stop = _work_stop(outcomes)
            if mode == MODE_NORMAL:
                if expected_stop is not None:
                    normal_stops.append((protocol.ONE_SHOT_STOP, expected_stop))
                if work_stop is not None:
                    normal_stops.append((STOP_WORK_CAP, work_stop))
                continue
            if expected_stop is not None:
                return _execution(
                    protocol.ONE_SHOT_STOP,
                    expected_stop,
                    prepared.sha256,
                    outcomes,
                )
            if work_stop is not None:
                return _execution(
                    STOP_WORK_CAP,
                    work_stop,
                    prepared.sha256,
                    outcomes,
                )

    work = _ledger(outcomes)
    if (
        work.producer_query_count != protocol.PRODUCER_QUERY_CAP
        or work.static_query_count != protocol.STATIC_QUERY_CAP
        or work.fitted_glif_step_count != protocol.FITTED_GLIF_STEP_CAP
        or work.plan_build_count != protocol.WORK_CAPS.plan_build_count
    ):
        normal_stops.append(
            (
                STOP_WORK_CAP,
                "complete one-shot work accounting differs from the frozen contract",
            )
        )
    if normal_stops:
        verdict = (
            STOP_WORK_CAP
            if any(status == STOP_WORK_CAP for status, _reason in normal_stops)
            else protocol.ONE_SHOT_STOP
        )
        reasons = tuple(dict.fromkeys(reason for _status, reason in normal_stops))
        return _execution(verdict, "; ".join(reasons), prepared.sha256, outcomes)
    return _execution(protocol.ONE_SHOT_PASS, None, prepared.sha256, outcomes)


def _result_artifact(
    prepared: _PreparedRun,
    authority: _ExecutionAuthority,
    execution: RunnerExecution,
) -> dict[str, object]:
    authority_receipt = _lookup_authority(authority)
    return {
        "schema": SCHEMA,
        "verdict": execution.verdict,
        "stop_reason": execution.stop_reason,
        "protocol": {
            "bank_commit": PROTOCOL_BANK_COMMIT,
            "source_sha256": PROTOCOL_SOURCE_SHA256,
            "protocol_sha256": PROTOCOL_ROOT_SHA256,
            "outcome_contract_sha256": PROTOCOL_OUTCOME_ROOT_SHA256,
        },
        "prespend_contract_sha256": PRESPEND_CONTRACT_SHA256,
        "prepared_sha256": prepared.sha256,
        "spend_marker": {
            "path": _display_path(SPENT_PATH),
            "sha256": authority_receipt.marker_sha256,
            "state": "PUBLIC_TRAIN_JOINT_ATTEMPT_SPENT",
            "exclusive_create": True,
            "flushed_and_fsynced_before_first_build": True,
        },
        "execution": execution,
        "scope": {
            "public_train_same_fit_finite_cue_replay": True,
            "old_panel_outcome_reads": 0,
            "dev_inputs": 0,
            "m0_inputs": 0,
            "c1_inputs": 0,
            "g0_inputs": 0,
            "protected_inputs": 0,
            "heldout_inputs": 0,
            "search_attempts": 0,
            "retries": 0,
            "cap_raises": 0,
            "row_substitutions": 0,
            "unseen_transfer_claim": False,
            "coding_capability_claim": False,
        },
    }


def run_one_shot() -> dict[str, object]:
    """Spend the exact two-row roster once and publish one immutable result."""

    prepared = _prepare_pre_spend()
    authority = _create_spend_authority(prepared)
    try:
        execution = _execute_prepared(prepared, authority, _REAL_BACKEND)
    except Exception as error:
        execution = _execution(
            protocol.ONE_SHOT_STOP,
            f"{type(error).__name__}: {error}",
            prepared.sha256,
            (),
        )
    artifact = _result_artifact(prepared, authority, execution)
    _write_json_exclusive(RESULT_PATH, artifact)
    return artifact


def main() -> None:
    if len(sys.argv) != 1:
        raise SystemExit("this one-shot runner accepts no options")
    print(json.dumps(_plain(run_one_shot()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "PRESPEND_CONTRACT_SHA256",
    "RESULT_PATH",
    "RUN_COMMAND",
    "SCHEMA",
    "SPENT_PATH",
    "RunnerExecution",
    "RunnerStop",
    "TreatmentOutcome",
    "WorkLedger",
    "run_one_shot",
]
