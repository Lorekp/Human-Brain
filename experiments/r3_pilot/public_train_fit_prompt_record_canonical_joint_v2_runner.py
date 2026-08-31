"""Exclusive one-shot runner for the fresh two-row V2 joint roster.

Importing this module spends nothing.  :func:`run_one_shot` authenticates the
exact roster plus swapped/forged preflights, durably spends the fresh V2 path,
then executes the frozen ten treatments.  Desired outputs remain private until
the bound-execution engine has authenticated an exact payload/frontier object.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import NamedTuple

from experiments.r3_pilot import (
    public_train_fit_prompt_record_canonical_joint_v2_bound_execution as engine,
)
from experiments.r3_pilot import (
    public_train_fit_prompt_record_canonical_joint_v2_protocol as protocol,
)

SCHEMA = "PUBLIC_TRAIN_FIT_PROMPT_RECORD_CANONICAL_JOINT_V2_RUNNER"
PRESPEND_READY = "PUBLIC_TRAIN_FIT_PROMPT_RECORD_CANONICAL_JOINT_V2_PRESPEND_READY"
STOP_PRESPEND = "STOP_PUBLIC_TRAIN_FIT_PROMPT_RECORD_CANONICAL_JOINT_V2_PRESPEND"
STOP_WORK_CAP = "STOP_PUBLIC_TRAIN_FIT_PROMPT_RECORD_CANONICAL_JOINT_V2_WORK_CAP"
STOP_ALREADY_SPENT = (
    "STOP_PUBLIC_TRAIN_FIT_PROMPT_RECORD_CANONICAL_JOINT_V2_ALREADY_SPENT"
)

PROTOCOL_BANK_COMMIT = engine.PROTOCOL_BANK_COMMIT
BOUND_ENGINE_SOURCE_SHA256 = (
    "18f9a6205a07a3b088288bbab4f08c100c5c2ab9b84dd411a04e7e96f628a4fe"
)
PROTOCOL_SOURCE_SHA256 = engine.PROTOCOL_SOURCE_SHA256
PROTOCOL_ROOT_SHA256 = engine.PROTOCOL_ROOT_SHA256
PROTOCOL_DEPENDENCY_SHA256 = engine.PROTOCOL_DEPENDENCY_SHA256
PROTOCOL_V1_IMMUTABLE_ROOT_SHA256 = (
    "6a8fceadf256124f47449d4b68b5dde3d24aa70f6ee31e946aa942338a332666"
)
PROTOCOL_ROSTER_ROOT_SHA256 = (
    "a0acd34f7dec36eead62eedca974d0a1f778388cb2a1add1c55e97e82771f9c1"
)
PROTOCOL_CONTROLS_ROOT_SHA256 = (
    "ba686377844a9e19ee8aeff3f11a7e909e60b45ff4a943fe9396a3edc87b99b8"
)
PROTOCOL_OUTCOME_ROOT_SHA256 = engine.PROTOCOL_OUTCOME_ROOT_SHA256
PROTOCOL_WORK_ROOT_SHA256 = engine.PROTOCOL_WORK_ROOT_SHA256
PROTOCOL_ORDER_ROOT_SHA256 = (
    "5598711be32152c1062452c0e002e5f2cd27a724ee74d5a000979254a71c8ba0"
)
PROTOCOL_SURFACE_ROOT_SHA256 = (
    "e8102cbbd1a476483356804f830f8d6eca92b8a4edd80932718d64a067ad7401"
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_RESULT_PATH = REPO_ROOT / protocol.FUTURE_RESULT_PATH
CANONICAL_SPENT_PATH = REPO_ROOT / protocol.FUTURE_SPENT_PATH
RESULT_PATH = CANONICAL_RESULT_PATH
SPENT_PATH = CANONICAL_SPENT_PATH
RUN_COMMAND = (
    r".\.venv\Scripts\python.exe -m "
    "experiments.r3_pilot."
    "public_train_fit_prompt_record_canonical_joint_v2_runner"
)

TREATMENT_ORDER = engine.TREATMENT_ORDER
_NAMESPACE = b"public-train-fit-prompt-record-canonical-joint-v2-runner\0"
_DEPENDENCY_SHA256S = {
    "experiments/r3_pilot/public_train_fit_prompt_record_canonical_joint_v2_protocol.py": (
        PROTOCOL_SOURCE_SHA256
    ),
    (
        "experiments/r3_pilot/public_train_fit_prompt_record_canonical_joint_v2_"
        "bound_execution.py"
    ): BOUND_ENGINE_SOURCE_SHA256,
}


def _plain(value: object) -> object:
    if isinstance(value, Path):
        return _display_path(value)
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if hasattr(value, "_asdict"):
        return _plain(value._asdict())
    if isinstance(value, (tuple, list)):
        return tuple(_plain(item) for item in value)
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


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


PRESPEND_CONTRACT = {
    "schema": SCHEMA,
    "protocol_bank_commit": PROTOCOL_BANK_COMMIT,
    "protocol_source_sha256": PROTOCOL_SOURCE_SHA256,
    "protocol_root_sha256": PROTOCOL_ROOT_SHA256,
    "bound_engine_source_sha256": BOUND_ENGINE_SOURCE_SHA256,
    "treatment_order": TREATMENT_ORDER,
    "row_order": (0, 1),
    "producer_query_cap": protocol.WORK_FORMULA.producer_query_cap,
    "static_query_cap": protocol.WORK_FORMULA.static_query_cap,
    "fitted_glif_step_cap": protocol.WORK_FORMULA.fitted_glif_step_cap,
    "plan_build_count": protocol.WORK_FORMULA.plan_build_count,
    "per_plan_work_caps": protocol.WORK_FORMULA.per_plan_build,
    "aggregate_plan_work_caps": protocol.WORK_FORMULA.aggregate_plan_builds,
    "result_path": _display_path(CANONICAL_RESULT_PATH),
    "spent_path": _display_path(CANONICAL_SPENT_PATH),
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


class _PreparedReceipt(NamedTuple):
    exact_preflight: protocol.FrozenPairingPreflight
    swapped_preflight: protocol.FrozenPairingPreflight
    forged_preflight: protocol.FrozenPairingPreflight
    dependency_sha256: str
    sha256: str


class _PreparedRun:
    """Opaque pre-spend state containing only zero-work preflights."""

    __slots__ = ()

    def __new__(cls) -> _PreparedRun:
        raise TypeError("prepared V2 runs require private runner authority")

    @property
    def sha256(self) -> str:
        return _lookup_prepared(self).sha256


class _AuthorityReceipt(NamedTuple):
    prepared: _PreparedRun
    executor_name: str
    real_executor: bool
    marker_sha256: str
    sha256: str


class _ExecutionAuthority:
    """Opaque authority bound to one exclusive durable spend marker."""

    __slots__ = ()

    def __new__(cls) -> _ExecutionAuthority:
        raise TypeError("V2 execution authorities require private runner authority")


class _TreatmentExecutor(NamedTuple):
    name: str
    real: bool
    call: Callable[
        [
            protocol.FrozenPairingPreflight,
            protocol.FrozenPublicTrainJointRow,
            int,
            str,
        ],
        engine.TreatmentOutcome,
    ]
    authenticate: Callable[[engine.TreatmentOutcome], object]


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
    treatment_outcomes: tuple[engine.TreatmentOutcome, ...]
    work: WorkLedger
    proof_materialization_count: int
    result_artifact_count_before_return: int
    sha256: str


_ISSUED_PREPARED: dict[int, tuple[_PreparedRun, _PreparedReceipt]] = {}
_ISSUED_AUTHORITIES: dict[
    int, tuple[_ExecutionAuthority, _AuthorityReceipt]
] = {}


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
        raise RunnerStop(
            STOP_PRESPEND, "execution authority is not an issued exact object"
        )
    return issued[1]


def _authenticate_dependencies() -> str:
    for relative, expected in _DEPENDENCY_SHA256S.items():
        if _file_sha256(REPO_ROOT / relative) != expected:
            raise RunnerStop(
                STOP_PRESPEND, f"banked V2 runner dependency changed: {relative}"
            )
    frozen = protocol.PROTOCOL
    if (
        frozen.protocol_sha256 != PROTOCOL_ROOT_SHA256
        or frozen.dependency_sha256 != PROTOCOL_DEPENDENCY_SHA256
        or frozen.v1_immutable_sha256 != PROTOCOL_V1_IMMUTABLE_ROOT_SHA256
        or frozen.roster_sha256 != PROTOCOL_ROSTER_ROOT_SHA256
        or frozen.controls_sha256 != PROTOCOL_CONTROLS_ROOT_SHA256
        or frozen.outcome_contract_sha256 != PROTOCOL_OUTCOME_ROOT_SHA256
        or frozen.work_formula_sha256 != PROTOCOL_WORK_ROOT_SHA256
        or frozen.one_shot_order_sha256 != PROTOCOL_ORDER_ROOT_SHA256
        or frozen.surface_boundary_sha256 != PROTOCOL_SURFACE_ROOT_SHA256
        or tuple(protocol.ROWS) != tuple(frozen.rows)
        or CANONICAL_RESULT_PATH.relative_to(REPO_ROOT).as_posix()
        != protocol.FUTURE_RESULT_PATH
        or CANONICAL_SPENT_PATH.relative_to(REPO_ROOT).as_posix()
        != protocol.FUTURE_SPENT_PATH
    ):
        raise RunnerStop(STOP_PRESPEND, "banked V2 protocol roots changed")
    engine_dependency = engine.authenticate_dependencies()
    return _digest(
        "dependencies",
        {
            "files": _DEPENDENCY_SHA256S,
            "engine_dependency_sha256": engine_dependency,
            "protocol_root_sha256": PROTOCOL_ROOT_SHA256,
        },
    )


def _assert_paths_absent() -> None:
    for label, path in (("spent marker", SPENT_PATH), ("result artifact", RESULT_PATH)):
        if path.exists():
            raise RunnerStop(STOP_ALREADY_SPENT, f"{label} already exists: {path}")


def _zero_preflight(receipt: protocol.FrozenPairingPreflight) -> bool:
    return (
        receipt.electrical_execution_count == 0
        and receipt.proof_materialization_count == 0
        and receipt.runtime_desired_read_count == 0
    )


def _prepare_pre_spend() -> _PreparedRun:
    """Authenticate exact, swapped, and forged rosters before any spend."""

    _assert_paths_absent()
    dependency_sha = _authenticate_dependencies()
    exact = protocol.preflight_pairing(protocol.ROWS)
    swapped = protocol.preflight_pairing(tuple(reversed(protocol.ROWS)))
    forged_row = object.__new__(protocol.FrozenPublicTrainJointRow)
    forged = protocol.preflight_pairing((protocol.ROWS[0], forged_row))
    if exact.status != protocol.PASS or not _zero_preflight(exact):
        raise RunnerStop(STOP_PRESPEND, "exact V2 atomic roster preflight stopped")
    if (
        swapped.status != protocol.STOP_PROMPT_IO_PAIRING_MISMATCH
        or not _zero_preflight(swapped)
    ):
        raise RunnerStop(STOP_PRESPEND, "swapped V2 roster did not stop atomically")
    if forged.status != protocol.STOP_ROW_CERTIFICATE or not _zero_preflight(forged):
        raise RunnerStop(STOP_PRESPEND, "forged V2 row did not stop atomically")
    record = {
        "status": PRESPEND_READY,
        "dependency_sha256": dependency_sha,
        "exact_preflight_sha256": exact.sha256,
        "swapped_preflight_sha256": swapped.sha256,
        "forged_preflight_sha256": forged.sha256,
        "electrical_execution_count": 0,
        "proof_materialization_count": 0,
        "runtime_desired_read_count": 0,
    }
    return _issue_prepared(
        _PreparedReceipt(
            exact,
            swapped,
            forged,
            dependency_sha,
            _digest("prepared-run", record),
        )
    )


def _write_json_exclusive(path: Path, payload: Mapping[str, object]) -> str:
    serialized = json.dumps(
        _plain(payload), indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise RunnerStop(STOP_ALREADY_SPENT, f"refusing to overwrite {path}") from error
    return _file_sha256(path)


def _create_spend_authority(
    prepared: _PreparedRun, executor: _TreatmentExecutor
) -> _ExecutionAuthority:
    prepared_receipt = _lookup_prepared(prepared)
    _assert_paths_absent()
    canonical_paths = (
        RESULT_PATH.resolve() == CANONICAL_RESULT_PATH.resolve()
        and SPENT_PATH.resolve() == CANONICAL_SPENT_PATH.resolve()
    )
    if executor.real and not canonical_paths:
        raise RunnerStop(
            STOP_PRESPEND,
            "real executor requires both exact canonical result/spend paths",
        )
    if not executor.real and (
        RESULT_PATH.resolve() == CANONICAL_RESULT_PATH.resolve()
        or SPENT_PATH.resolve() == CANONICAL_SPENT_PATH.resolve()
    ):
        raise RunnerStop(
            STOP_PRESPEND,
            "non-real executor may write only isolated focused-test paths",
        )
    marker = {
        "schema": SCHEMA,
        "state": "PUBLIC_TRAIN_JOINT_V2_ATTEMPT_SPENT",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "protocol_bank_commit": PROTOCOL_BANK_COMMIT,
        "protocol_root_sha256": PROTOCOL_ROOT_SHA256,
        "prepared_sha256": prepared_receipt.sha256,
        "runner_source_sha256": _file_sha256(Path(__file__)),
        "executor_name": executor.name,
        "real_electrical_executor": executor.real,
        "result_path": _display_path(RESULT_PATH),
        "no_retry": True,
    }
    marker_sha = _write_json_exclusive(SPENT_PATH, marker)
    if RESULT_PATH.exists():
        raise RunnerStop(
            STOP_ALREADY_SPENT,
            "result appeared while the spend marker was created; marker retained",
        )
    values = (prepared_receipt.sha256, executor.name, executor.real, marker_sha)
    return _issue_authority(
        _AuthorityReceipt(
            prepared,
            executor.name,
            executor.real,
            marker_sha,
            _digest("execution-authority", values),
        )
    )


def _authenticate_authority(
    prepared: _PreparedRun,
    authority: _ExecutionAuthority,
    executor: _TreatmentExecutor,
) -> _AuthorityReceipt:
    _lookup_prepared(prepared)
    receipt = _lookup_authority(authority)
    if (
        receipt.prepared is not prepared
        or receipt.executor_name != executor.name
        or receipt.real_executor is not executor.real
        or not SPENT_PATH.is_file()
        or _file_sha256(SPENT_PATH) != receipt.marker_sha256
    ):
        raise RunnerStop(STOP_PRESPEND, "durable V2 execution authority changed")
    return receipt


def _private_desired(
    binding: engine.ExactPayloadFrontierBinding, *, wrong: bool
) -> tuple[int, ...]:
    """Open desired only after the engine authenticates an exact binding."""

    receipt = engine.authenticate_binding(binding)
    if not binding.requires_desired:
        raise RunnerStop(STOP_PRESPEND, "non-READY binding requested desired authority")
    desired = tuple(protocol._lookup_row(receipt.row).selection.desired)
    if not wrong:
        return desired
    shifted = desired[-1:] + desired[:-1]
    if shifted == desired:
        shifted = tuple(value + index + 1 for index, value in enumerate(desired))
    if shifted == desired:
        raise RuntimeError("frozen desired permutation failed to change the target")
    return shifted


def _real_treatment(
    preflight: protocol.FrozenPairingPreflight,
    row: protocol.FrozenPublicTrainJointRow,
    row_ordinal: int,
    mode: str,
) -> engine.TreatmentOutcome:
    binding = engine.bind_treatment(preflight, row, row_ordinal, mode)
    engine.authenticate_binding(binding)
    if not binding.requires_desired:
        return engine.finish_bound(binding)
    desired = _private_desired(
        binding, wrong=mode == engine.MODE_WRONG_DESIRED
    )
    return engine.select_bound(binding, desired)


_REAL_EXECUTOR = _TreatmentExecutor(
    "banked-v2-bound-execution",
    True,
    _real_treatment,
    engine.authenticate_outcome,
)


def _complete_static_queries(source_count: int) -> int:
    return source_count * source_count + 18 * source_count


def _expected(outcome: engine.TreatmentOutcome) -> str | None:
    geometry = protocol.CANDIDATE_GEOMETRY_RECEIPTS[outcome.row_ordinal]
    if (
        outcome.source_candidate_event_count != geometry.total_candidate_count
        or outcome.geometry_sha256 != geometry.sha256
        or outcome.producer_build_attempts != 1
    ):
        return f"{outcome.mode} row {outcome.row_ordinal} changed candidate geometry"
    n = outcome.source_event_count
    if outcome.mode == engine.MODE_PRODUCER_OFF:
        valid = (
            outcome.build_status == engine.ZERO_SELECTED_SOURCE_EVENTS
            and outcome.source_event_status == engine.prompt_producer.STOP_ZERO
            and n == 0
            and outcome.occurrence_status is None
            and outcome.occurrence_inference_attempts == 0
            and outcome.plan_build_attempts == 0
            and outcome.runtime_desired_reads == 0
            and outcome.static_query_count == 0
            and outcome.exact_proof_count == 0
            and outcome.witness_count == 0
        )
    elif outcome.mode == engine.MODE_STATIC_OFF:
        valid = (
            outcome.build_status == engine.occurrence_static.STOP_ZERO
            and outcome.source_event_status == engine.prompt_producer.STOP_UNSUPPORTED
            and n in protocol.SUPPORTED_SOURCE_EVENT_COUNTS
            and outcome.occurrence_status == engine.occurrence_static.STOP_ZERO
            and outcome.occurrence_count == 0
            and outcome.occurrence_inference_attempts == 1
            and outcome.plan_build_attempts == 0
            and outcome.runtime_desired_reads == 0
            and outcome.static_query_count == n
            and outcome.exact_proof_count == 0
            and outcome.witness_count == 0
        )
    elif outcome.mode == engine.MODE_CONVERTER_OFF:
        valid = (
            outcome.build_status == engine.STOP_SOURCE_CERTIFICATE
            and outcome.source_event_status == engine.prompt_producer.READY
            and n in protocol.SUPPORTED_SOURCE_EVENT_COUNTS
            and outcome.occurrence_status == engine.occurrence_static.READY
            and outcome.occurrence_count == n
            and outcome.occurrence_inference_attempts == 2
            and outcome.plan_build_attempts == 0
            and outcome.runtime_desired_reads == 0
            and outcome.static_query_count == 2 * _complete_static_queries(n)
            and outcome.exact_proof_count == 0
            and outcome.witness_count == 0
        )
    elif outcome.mode == engine.MODE_WRONG_DESIRED:
        valid = (
            outcome.build_status == engine.READY
            and outcome.source_event_status == engine.prompt_producer.READY
            and n in protocol.SUPPORTED_SOURCE_EVENT_COUNTS
            and outcome.occurrence_status == engine.occurrence_static.READY
            and outcome.occurrence_count == n
            and outcome.occurrence_inference_attempts == 2
            and outcome.plan_build_attempts == 1
            and outcome.runtime_desired_reads == protocol.VISIBLE_CASE_COUNT
            and outcome.selection_status
            in (engine.STOP_NO_EXACT, engine.STOP_AUTHORIZED_ECLASS_MISMATCH)
            and outcome.public_status
            in (engine.STOP_NO_EXACT, protocol.STOP_PROMPT_IO_PAIRING_MISMATCH)
            and outcome.static_query_count == 2 * _complete_static_queries(n)
            and outcome.exact_proof_count == 0
            and outcome.witness_count == 0
            and outcome.selected_proof_forest_sha256 is None
        )
    elif outcome.mode == engine.MODE_NORMAL:
        common = (
            outcome.build_status == engine.READY
            and outcome.source_event_status == engine.prompt_producer.READY
            and n in protocol.SUPPORTED_SOURCE_EVENT_COUNTS
            and outcome.occurrence_status == engine.occurrence_static.READY
            and outcome.occurrence_count == n
            and outcome.occurrence_inference_attempts == 2
            and outcome.plan_build_attempts == 1
            and outcome.runtime_desired_reads == protocol.VISIBLE_CASE_COUNT
            and outcome.static_query_count == 2 * _complete_static_queries(n)
        )
        selected = (
            outcome.public_status == engine.SELECTED
            and outcome.selection_status == engine.SELECTED
            and outcome.exact_proof_count == 1
            and outcome.witness_count >= 1
            and outcome.selected_proof_forest_sha256 is not None
        )
        stopped = (
            outcome.public_status
            in (engine.STOP_NO_EXACT, protocol.STOP_PROMPT_IO_PAIRING_MISMATCH)
            and outcome.selection_status
            in (engine.STOP_NO_EXACT, engine.STOP_AUTHORIZED_ECLASS_MISMATCH)
            and outcome.exact_proof_count == 0
            and outcome.witness_count == 0
            and outcome.selected_proof_forest_sha256 is None
        )
        valid = common and (selected or stopped)
    else:
        return f"unknown frozen treatment mode {outcome.mode}"
    if not valid:
        return f"{outcome.mode} row {outcome.row_ordinal} violated its frozen outcome"
    return None


def _ledger(outcomes: Sequence[engine.TreatmentOutcome]) -> WorkLedger:
    producer = sum(item.source_candidate_event_count for item in outcomes)
    static = sum(item.static_query_count for item in outcomes)
    plan_builds = sum(item.plan_build_attempts for item in outcomes)
    relation = sum(item.relation_work_units for item in outcomes)
    predictions = sum(item.physical_predictions for item in outcomes)
    races = sum(item.physical_races for item in outcomes)
    fitted = (producer + static) * protocol.WORK_FORMULA.fitted_glif_steps_per_query
    values = (producer, static, fitted, plan_builds, relation, predictions, races)
    return WorkLedger(*values, _digest("work-ledger", values))


def _work_stop(outcomes: Sequence[engine.TreatmentOutcome]) -> str | None:
    per_plan = protocol.WORK_FORMULA.per_plan_build
    for outcome in outcomes:
        if (
            outcome.relation_work_units > per_plan[0]
            or outcome.physical_predictions > per_plan[1]
            or outcome.physical_races > per_plan[2]
        ):
            return f"{outcome.mode} row {outcome.row_ordinal} exceeded a plan work cap"
    ledger = _ledger(outcomes)
    aggregate = protocol.WORK_FORMULA.aggregate_plan_builds
    if (
        ledger.producer_query_count > protocol.WORK_FORMULA.producer_query_cap
        or ledger.static_query_count > protocol.WORK_FORMULA.static_query_cap
        or ledger.fitted_glif_step_count > protocol.WORK_FORMULA.fitted_glif_step_cap
        or ledger.plan_build_count > protocol.WORK_FORMULA.plan_build_count
        or ledger.relation_work_units > aggregate[0]
        or ledger.physical_predictions > aggregate[1]
        or ledger.physical_races > aggregate[2]
    ):
        return "aggregate V2 one-shot work cap exceeded"
    return None


def _execution(
    verdict: str,
    reason: str | None,
    prepared_sha256: str,
    outcomes: Sequence[engine.TreatmentOutcome],
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


def _execute_prepared(
    prepared: _PreparedRun,
    authority: _ExecutionAuthority,
    executor: _TreatmentExecutor,
) -> RunnerExecution:
    """Execute all ten calls; a normal row STOP never suppresses row two."""

    prepared_receipt = _lookup_prepared(prepared)
    _authenticate_authority(prepared, authority, executor)
    outcomes: list[engine.TreatmentOutcome] = []
    reasons: list[str] = []
    for mode in TREATMENT_ORDER:
        for row_ordinal, row in enumerate(protocol.ROWS):
            _authenticate_authority(prepared, authority, executor)
            try:
                outcome = executor.call(
                    prepared_receipt.exact_preflight,
                    row,
                    row_ordinal,
                    mode,
                )
                executor.authenticate(outcome)
                if (
                    type(outcome) is not engine.TreatmentOutcome
                    or outcome.mode != mode
                    or outcome.row_ordinal != row_ordinal
                    or type(outcome.sha256) is not str
                    or len(outcome.sha256) != 64
                ):
                    raise TypeError("executor returned the wrong treatment receipt")
            except Exception as error:
                reasons.append(
                    f"{mode} row {row_ordinal} execution failed: "
                    f"{type(error).__name__}: {error}"
                )
                return _execution(
                    protocol.ONE_SHOT_STOP,
                    "; ".join(reasons),
                    prepared.sha256,
                    outcomes,
                )
            outcomes.append(outcome)
            expected_stop = _expected(outcome)
            work_reason = _work_stop(outcomes)
            if expected_stop is not None or work_reason is not None:
                immediate = tuple(
                    reason
                    for reason in (expected_stop, work_reason)
                    if reason is not None
                )
                return _execution(
                    STOP_WORK_CAP if work_reason is not None else protocol.ONE_SHOT_STOP,
                    "; ".join(immediate),
                    prepared.sha256,
                    outcomes,
                )
            if mode == engine.MODE_NORMAL and outcome.public_status != engine.SELECTED:
                reasons.append(
                    f"normal row {row_ordinal} did not select an exact proof"
                )

    for row_ordinal in range(len(protocol.ROWS)):
        counts = {
            item.source_event_count
            for item in outcomes
            if item.row_ordinal == row_ordinal
            and item.mode != engine.MODE_PRODUCER_OFF
        }
        if len(counts) != 1:
            reasons.append(f"row {row_ordinal} source bundle changed across treatments")
    work = _ledger(outcomes)
    if work.producer_query_count != protocol.WORK_FORMULA.producer_query_cap:
        reasons.append("complete producer work differs from frozen candidate geometry")
    if work.plan_build_count != protocol.WORK_FORMULA.plan_build_count:
        reasons.append("complete plan build count differs from the frozen contract")
    work_reason = _work_stop(outcomes)
    if work_reason is not None:
        reasons.append(work_reason)
    unique_reasons = tuple(dict.fromkeys(reasons))
    if work_reason is not None:
        return _execution(
            STOP_WORK_CAP, "; ".join(unique_reasons), prepared.sha256, outcomes
        )
    if unique_reasons:
        return _execution(
            protocol.ONE_SHOT_STOP,
            "; ".join(unique_reasons),
            prepared.sha256,
            outcomes,
        )
    return _execution(protocol.ONE_SHOT_PASS, None, prepared.sha256, outcomes)


def _result_artifact(
    prepared: _PreparedRun,
    authority: _ExecutionAuthority,
    execution: RunnerExecution,
) -> dict[str, object]:
    prepared_receipt = _lookup_prepared(prepared)
    authority_receipt = _lookup_authority(authority)
    return {
        "schema": SCHEMA,
        "verdict": execution.verdict,
        "stop_reason": execution.stop_reason,
        "protocol": {
            "bank_commit": PROTOCOL_BANK_COMMIT,
            "source_sha256": PROTOCOL_SOURCE_SHA256,
            "protocol_sha256": PROTOCOL_ROOT_SHA256,
            "dependency_sha256": PROTOCOL_DEPENDENCY_SHA256,
            "roster_sha256": PROTOCOL_ROSTER_ROOT_SHA256,
            "controls_sha256": PROTOCOL_CONTROLS_ROOT_SHA256,
            "outcome_contract_sha256": PROTOCOL_OUTCOME_ROOT_SHA256,
            "work_formula_sha256": PROTOCOL_WORK_ROOT_SHA256,
            "one_shot_order_sha256": PROTOCOL_ORDER_ROOT_SHA256,
        },
        "bound_engine_source_sha256": BOUND_ENGINE_SOURCE_SHA256,
        "prespend_contract_sha256": PRESPEND_CONTRACT_SHA256,
        "prepared_sha256": prepared.sha256,
        "preflights": {
            "exact_sha256": prepared_receipt.exact_preflight.sha256,
            "swapped_sha256": prepared_receipt.swapped_preflight.sha256,
            "forged_sha256": prepared_receipt.forged_preflight.sha256,
            "electrical_execution_count": 0,
            "proof_materialization_count": 0,
            "runtime_desired_read_count": 0,
        },
        "spend_marker": {
            "path": _display_path(SPENT_PATH),
            "sha256": authority_receipt.marker_sha256,
            "state": "PUBLIC_TRAIN_JOINT_V2_ATTEMPT_SPENT",
            "exclusive_create": True,
            "flushed_and_fsynced_before_first_treatment": True,
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


def _run_with_executor(executor: _TreatmentExecutor) -> dict[str, object]:
    """Focused internal seam; non-real executors require noncanonical paths."""

    prepared = _prepare_pre_spend()
    authority = _create_spend_authority(prepared, executor)
    try:
        execution = _execute_prepared(prepared, authority, executor)
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


def run_one_shot() -> dict[str, object]:
    """Spend the exact fresh V2 roster once and publish one immutable result."""

    return _run_with_executor(_REAL_EXECUTOR)


def main() -> None:
    if len(sys.argv) != 1:
        raise SystemExit("this one-shot V2 runner accepts no options")
    print(json.dumps(_plain(run_one_shot()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


CLAIM_BOUNDARY = (
    "one exclusive fresh two-row public-TRAIN same-FIT exact-bound execution "
    "with causal controls and private desired opening only; no retries, unseen "
    "prompt/family transfer, language understanding, coding capability, DEV, "
    "heldout, M0, C1, G0, or protected evidence"
)


__all__ = [
    "CLAIM_BOUNDARY",
    "PRESPEND_CONTRACT_SHA256",
    "RESULT_PATH",
    "RUN_COMMAND",
    "SCHEMA",
    "SPENT_PATH",
    "RunnerExecution",
    "RunnerStop",
    "WorkLedger",
    "run_one_shot",
]
