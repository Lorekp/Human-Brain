"""Exclusive one-shot runner for the fresh two-row V3 joint roster.

Importing this module spends nothing.  :func:`run_one_shot` authenticates the
exact V3 protocol and engine, durably spends both canonical artifact paths,
then binds each exact row once.  A scientific STOP is retained and the next row
still runs; desired values open only behind an authenticated READY binding.
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
    public_train_fit_prompt_record_canonical_joint_v3_bound_execution as engine,
)
from experiments.r3_pilot import (
    public_train_fit_prompt_record_canonical_joint_v3_protocol as protocol,
)

SCHEMA = "PUBLIC_TRAIN_FIT_PROMPT_RECORD_CANONICAL_JOINT_V3_RUNNER"
PRESPEND_READY = "PUBLIC_TRAIN_FIT_PROMPT_RECORD_CANONICAL_JOINT_V3_PRESPEND_READY"
STOP_PRESPEND = "STOP_PUBLIC_TRAIN_FIT_PROMPT_RECORD_CANONICAL_JOINT_V3_PRESPEND"
STOP_WORK_CAP = "STOP_PUBLIC_TRAIN_FIT_PROMPT_RECORD_CANONICAL_JOINT_V3_WORK_CAP"
STOP_ALREADY_SPENT = (
    "STOP_PUBLIC_TRAIN_FIT_PROMPT_RECORD_CANONICAL_JOINT_V3_ALREADY_SPENT"
)

PROTOCOL_BANK_COMMIT = engine.PROTOCOL_BANK_COMMIT
BOUND_ENGINE_BANK_COMMIT = "70a5e1f68373aeaf474e041f4551a84fc30e7488"
BOUND_ENGINE_SOURCE_SHA256 = (
    "70dbaf9acaab42c9084276fc89115cdfd57172322aa06ba07cde5ebbec078f73"
)
PROTOCOL_SOURCE_SHA256 = engine.PROTOCOL_SOURCE_SHA256
PROTOCOL_ROOT_SHA256 = engine.PROTOCOL_ROOT_SHA256
PROTOCOL_DEPENDENCY_SHA256 = (
    "4af097517d8358cc092f97375643113c2765553974c93424225c0ecf3f97ed6d"
)
PROTOCOL_TERMINAL_COHORTS_SHA256 = (
    "31f210a8c28152d24699fa75731d443a7c079b58281ced711f9419db0305144e"
)
PROTOCOL_ROSTER_SHA256 = (
    "b6758adc818038c1cbbc436cb702a53e21bd48b63cb58ad9250786424c03e1e7"
)
PROTOCOL_EXECUTION_CONTRACT_SHA256 = (
    "f6a1bbabafdd04db82c9a2a8148923d8ad0614f6160be873e86249123ca7f185"
)
PROTOCOL_WORK_FORMULA_SHA256 = (
    "5a207128248125aee77fae2ce73c1395b36bd153c68278bec28b15e87e738339"
)
PROTOCOL_ORDER_SHA256 = (
    "6a36f16a0232cad04ae6a2e58c988404060dcf3edcde4e8bc6f82e4b2ec61060"
)
PROTOCOL_SURFACE_SHA256 = (
    "f3247fdc1ab000f0ad99532792e3454ad79255b0d6bc2c1bca488a6bd6487211"
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_RESULT_PATH = REPO_ROOT / protocol.FUTURE_RESULT_PATH
CANONICAL_SPENT_PATH = REPO_ROOT / protocol.FUTURE_SPENT_PATH
RESULT_PATH = CANONICAL_RESULT_PATH
SPENT_PATH = CANONICAL_SPENT_PATH
RUN_COMMAND = (
    r".\.venv\Scripts\python.exe -m "
    "experiments.r3_pilot.public_train_fit_prompt_record_canonical_joint_v3_runner"
)

_NAMESPACE = b"public-train-fit-prompt-record-canonical-joint-v3-runner\0"
_DEPENDENCY_SHA256S = {
    "experiments/r3_pilot/public_train_fit_prompt_record_canonical_joint_v3_protocol.py": (
        PROTOCOL_SOURCE_SHA256
    ),
    (
        "experiments/r3_pilot/public_train_fit_prompt_record_canonical_joint_v3_"
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
    "bound_engine_bank_commit": BOUND_ENGINE_BANK_COMMIT,
    "bound_engine_source_sha256": BOUND_ENGINE_SOURCE_SHA256,
    "row_order": (0, 1),
    "row_bind_count": (1, 1),
    "selection_order_after_ready": engine.SELECTION_ORDER,
    "candidate_counts": protocol.WORK_FORMULA.per_row_authenticated_candidate_counts,
    "selector_passes_per_row": protocol.WORK_FORMULA.selector_passes_per_row,
    "static_passes_per_row": protocol.WORK_FORMULA.static_passes_per_row,
    "source_query_cap": protocol.WORK_FORMULA.source_query_cap,
    "static_query_cap": protocol.WORK_FORMULA.static_query_cap,
    "fitted_glif_step_cap": protocol.WORK_FORMULA.fitted_glif_step_cap,
    "selected_count_bounds": protocol.WORK_FORMULA.per_row_selected_count_bounds,
    "result_path": _display_path(CANONICAL_RESULT_PATH),
    "spent_path": _display_path(CANONICAL_SPENT_PATH),
    "retry_allowed": False,
    "search_allowed": False,
    "cap_raise_allowed": False,
    "row_substitution_allowed": False,
    "static_stop_suppresses_later_rows": False,
}
PRESPEND_CONTRACT_SHA256 = _digest("prespend-contract", PRESPEND_CONTRACT)


class RunnerStop(RuntimeError):
    """Fail-closed runner integrity or work error."""

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
    """Opaque zero-work pre-spend state."""

    __slots__ = ()

    def __new__(cls) -> _PreparedRun:
        raise TypeError("prepared V3 runs require private runner authority")

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
        raise TypeError("V3 execution authorities require private runner authority")


class _RowExecutor(NamedTuple):
    name: str
    real: bool
    call: Callable[
        [protocol.FrozenPairingPreflight, protocol.FrozenPublicTrainJointRow, int],
        Sequence[engine.RowTreatmentOutcome],
    ]
    authenticate: Callable[[engine.RowTreatmentOutcome], object]


class WorkLedger(NamedTuple):
    source_query_count: int
    source_fitted_glif_step_count: int
    selected_source_counts: tuple[int, ...]
    static_off_query_count: int
    static_on_query_count: int
    static_query_count: int
    static_query_dynamic_cap: int
    static_fitted_glif_step_count: int
    fitted_glif_step_count: int
    converter_build_count: int
    plan_build_count: int
    relation_completed_units: int
    relation_attempted_units: int
    relation_limit: int
    physical_predictions: int
    physical_races: int
    proof_materialization_count: int
    sha256: str


class RunnerExecution(NamedTuple):
    verdict: str
    stop_reason: str | None
    prepared_sha256: str
    row_outcomes: tuple[tuple[engine.RowTreatmentOutcome, ...], ...]
    work: WorkLedger
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
        raise RunnerStop(STOP_PRESPEND, "prepared V3 run has the wrong type")
    issued = _ISSUED_PREPARED.get(id(prepared))
    if issued is None or issued[0] is not prepared:
        raise RunnerStop(STOP_PRESPEND, "prepared V3 run is not an issued object")
    return issued[1]


def _issue_authority(receipt: _AuthorityReceipt) -> _ExecutionAuthority:
    authority = object.__new__(_ExecutionAuthority)
    _ISSUED_AUTHORITIES[id(authority)] = (authority, receipt)
    return authority


def _lookup_authority(authority: object) -> _AuthorityReceipt:
    if type(authority) is not _ExecutionAuthority:
        raise RunnerStop(STOP_PRESPEND, "V3 execution authority has the wrong type")
    issued = _ISSUED_AUTHORITIES.get(id(authority))
    if issued is None or issued[0] is not authority:
        raise RunnerStop(STOP_PRESPEND, "V3 execution authority is not issued")
    return issued[1]


def _authenticate_dependencies() -> str:
    for relative, expected in _DEPENDENCY_SHA256S.items():
        if _file_sha256(REPO_ROOT / relative) != expected:
            raise RunnerStop(
                STOP_PRESPEND, f"banked V3 runner dependency changed: {relative}"
            )
    frozen = protocol.PROTOCOL
    if (
        frozen.protocol_sha256 != PROTOCOL_ROOT_SHA256
        or frozen.dependency_sha256 != PROTOCOL_DEPENDENCY_SHA256
        or frozen.terminal_cohorts_sha256 != PROTOCOL_TERMINAL_COHORTS_SHA256
        or frozen.roster_sha256 != PROTOCOL_ROSTER_SHA256
        or frozen.execution_contract_sha256
        != PROTOCOL_EXECUTION_CONTRACT_SHA256
        or frozen.work_formula_sha256 != PROTOCOL_WORK_FORMULA_SHA256
        or frozen.one_shot_order_sha256 != PROTOCOL_ORDER_SHA256
        or frozen.surface_boundary_sha256 != PROTOCOL_SURFACE_SHA256
        or tuple(protocol.ROWS) != tuple(frozen.rows)
        or CANONICAL_RESULT_PATH.relative_to(REPO_ROOT).as_posix()
        != protocol.FUTURE_RESULT_PATH
        or CANONICAL_SPENT_PATH.relative_to(REPO_ROOT).as_posix()
        != protocol.FUTURE_SPENT_PATH
        or engine.PROTOCOL_BANK_COMMIT != PROTOCOL_BANK_COMMIT
    ):
        raise RunnerStop(STOP_PRESPEND, "banked V3 protocol roots changed")
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
    """Authenticate exact/order/forgery controls with zero physical work."""

    _assert_paths_absent()
    dependency_sha = _authenticate_dependencies()
    exact = protocol.preflight_pairing(protocol.ROWS)
    swapped = protocol.preflight_pairing(tuple(reversed(protocol.ROWS)))
    forged_row = object.__new__(protocol.FrozenPublicTrainJointRow)
    forged = protocol.preflight_pairing((protocol.ROWS[0], forged_row))
    if exact.status != protocol.PASS or not _zero_preflight(exact):
        raise RunnerStop(STOP_PRESPEND, "exact V3 atomic roster preflight stopped")
    if (
        swapped.status != protocol.STOP_PROMPT_IO_PAIRING_MISMATCH
        or not _zero_preflight(swapped)
    ):
        raise RunnerStop(STOP_PRESPEND, "swapped V3 roster did not stop atomically")
    if forged.status != protocol.STOP_ROW_CERTIFICATE or not _zero_preflight(forged):
        raise RunnerStop(STOP_PRESPEND, "forged V3 row did not stop atomically")
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
    prepared: _PreparedRun, executor: _RowExecutor
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
            "real executor requires both exact canonical V3 result/spend paths",
        )
    if not executor.real and (
        RESULT_PATH.resolve() == CANONICAL_RESULT_PATH.resolve()
        or SPENT_PATH.resolve() == CANONICAL_SPENT_PATH.resolve()
    ):
        raise RunnerStop(
            STOP_PRESPEND,
            "non-real executor may not target either canonical V3 path",
        )
    marker = {
        "schema": SCHEMA,
        "state": "PUBLIC_TRAIN_JOINT_V3_ATTEMPT_SPENT",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "protocol_bank_commit": PROTOCOL_BANK_COMMIT,
        "protocol_root_sha256": PROTOCOL_ROOT_SHA256,
        "bound_engine_bank_commit": BOUND_ENGINE_BANK_COMMIT,
        "bound_engine_source_sha256": BOUND_ENGINE_SOURCE_SHA256,
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
            "result appeared while V3 spend marker was created; marker retained",
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
    executor: _RowExecutor,
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
        raise RunnerStop(STOP_PRESPEND, "durable V3 execution authority changed")
    return receipt


def _private_desired(
    binding: engine.ExactRowExecutionBinding, *, wrong: bool
) -> tuple[int, ...]:
    """Open desired only after exact READY binding authentication."""

    receipt = engine.authenticate_binding(binding)
    if not binding.requires_desired:
        raise RunnerStop(STOP_PRESPEND, "non-READY V3 binding requested desired")
    desired = tuple(protocol._lookup_row(receipt.row).selection.desired)
    if not wrong:
        return desired
    shifted = desired[-1:] + desired[:-1]
    if shifted == desired:
        shifted = tuple(value + index + 1 for index, value in enumerate(desired))
    if shifted == desired:
        raise RuntimeError("frozen V3 desired permutation did not change the target")
    return shifted


def _real_row(
    preflight: protocol.FrozenPairingPreflight,
    row: protocol.FrozenPublicTrainJointRow,
    row_ordinal: int,
) -> tuple[engine.RowTreatmentOutcome, ...]:
    binding = engine.bind_row(preflight, row, row_ordinal)
    engine.authenticate_binding(binding)
    if not binding.requires_desired:
        return (engine.finish_bound(binding),)
    wrong = engine.select_bound(
        binding,
        _private_desired(binding, wrong=True),
        mode=engine.MODE_WRONG_DESIRED,
    )
    normal = engine.select_bound(
        binding,
        _private_desired(binding, wrong=False),
        mode=engine.MODE_NORMAL,
    )
    return wrong, normal


_REAL_EXECUTOR = _RowExecutor(
    "canonical-v3-exact-row-executor", True, _real_row, engine.authenticate_outcome
)


def _valid_sha256(value: object, *, optional: bool = False) -> bool:
    if optional and value is None:
        return True
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _complete_static_queries(source_count: int) -> int:
    return source_count + 6 * (source_count * source_count + 18 * source_count)


def _stage_key(outcome: engine.RowTreatmentOutcome) -> tuple[object, ...]:
    return (
        outcome.row_ordinal,
        outcome.build_status,
        outcome.source_gain0_status,
        outcome.source_gain1_status,
        outcome.static_off_status,
        outcome.static_on_status,
        outcome.source_candidate_event_count,
        outcome.source_selected_event_count,
        outcome.selector_pass_count,
        outcome.source_query_count,
        outcome.source_fitted_glif_step_count,
        outcome.source_compiled_kernel_call_count,
        outcome.static_pass_count,
        outcome.static_off_query_count,
        outcome.static_on_query_count,
        outcome.static_query_count,
        outcome.static_fitted_glif_step_count,
        outcome.converter_control_status,
        outcome.converter_control_plan_build_attempts,
        outcome.converter_build_attempts,
        outcome.plan_build_attempts,
        outcome.dependency_sha256,
        outcome.payload_sha256,
        outcome.geometry_sha256,
        outcome.source_certificate_sha256s,
        outcome.source_request_sha256s,
        outcome.source_candidate_event_root_sha256s,
        outcome.source_selected_event_root_sha256s,
        outcome.source_work_sha256s,
        outcome.static_certificate_sha256s,
        outcome.static_request_sha256s,
        outcome.static_view_sha256s,
        outcome.static_work_sha256,
        outcome.converter_control_frontier_sha256,
        outcome.frontier_sha256,
        outcome.binding_sha256,
    )


def _validate_roots(outcome: engine.RowTreatmentOutcome) -> bool:
    required_pairs = (
        outcome.source_certificate_sha256s,
        outcome.source_request_sha256s,
        outcome.source_candidate_event_root_sha256s,
        outcome.source_selected_event_root_sha256s,
        outcome.source_work_sha256s,
        outcome.static_certificate_sha256s,
        outcome.static_request_sha256s,
    )
    return (
        all(
            len(values) == 2 and all(_valid_sha256(item) for item in values)
            for values in required_pairs
        )
        and len(outcome.static_view_sha256s) == 2
        and all(_valid_sha256(item, optional=True) for item in outcome.static_view_sha256s)
        and all(
            _valid_sha256(item, optional=optional)
            for item, optional in (
                (outcome.dependency_sha256, False),
                (outcome.payload_sha256, False),
                (outcome.geometry_sha256, False),
                (outcome.static_work_sha256, False),
                (outcome.converter_control_frontier_sha256, True),
                (outcome.frontier_sha256, True),
                (outcome.binding_sha256, False),
                (outcome.frontier_work_sha256, True),
                (outcome.sha256, False),
            )
        )
    )


def _validate_row_outcomes(
    row_ordinal: int,
    supplied: Sequence[engine.RowTreatmentOutcome],
) -> tuple[tuple[engine.RowTreatmentOutcome, ...], str | None]:
    outcomes = tuple(supplied)
    if not outcomes:
        raise RunnerStop(STOP_PRESPEND, f"row {row_ordinal} returned no outcome")
    for outcome in outcomes:
        if (
            type(outcome) is not engine.RowTreatmentOutcome
            or outcome.row_ordinal != row_ordinal
            or not _validate_roots(outcome)
        ):
            raise RunnerStop(
                STOP_PRESPEND, f"row {row_ordinal} returned an invalid exact receipt"
            )
    if any(_stage_key(item) != _stage_key(outcomes[0]) for item in outcomes[1:]):
        raise RunnerStop(
            STOP_PRESPEND, f"row {row_ordinal} reran or changed its bound stage"
        )

    first = outcomes[0]
    geometry = protocol.CANDIDATE_GEOMETRY_RECEIPTS[row_ordinal]
    n = first.source_selected_event_count
    complete_static = _complete_static_queries(n)
    if (
        first.geometry_sha256 != geometry.sha256
        or first.source_candidate_event_count != geometry.total_candidate_count
        or first.source_gain0_status != engine.source_static.SOURCE_ZERO
        or first.source_gain1_status != engine.source_static.SOURCE_SELECTED
        or first.selector_pass_count != 2
        or not 1 <= n <= geometry.total_candidate_count
        or first.source_query_count != 2 * geometry.total_candidate_count
        or first.source_fitted_glif_step_count != first.source_query_count * 30
        or first.source_compiled_kernel_call_count != 2
        or first.static_pass_count != 2
        or first.static_off_status != engine.occurrence_static.STOP_ZERO
        or first.static_off_query_count != n
        or first.static_on_query_count not in (n, complete_static)
        or first.static_query_count
        != first.static_off_query_count + first.static_on_query_count
        or first.static_fitted_glif_step_count != first.static_query_count * 30
    ):
        raise RunnerStop(
            STOP_PRESPEND, f"row {row_ordinal} violated source/static work authority"
        )

    if first.static_on_status != engine.occurrence_static.READY:
        valid_stop = (
            len(outcomes) == 1
            and first.build_status == first.static_on_status
            and first.public_status == first.static_on_status
            and first.selection_mode is None
            and first.selection_status is None
            and first.converter_control_status is None
            and first.converter_control_plan_build_attempts == 0
            and first.converter_build_attempts == 0
            and first.plan_build_attempts == 0
            and first.runtime_desired_reads == 0
            and first.exact_proof_count == 0
            and first.witness_count == 0
            and first.selected_proof_forest_sha256 is None
            and first.relation_completed_units == 0
            and first.relation_attempted_units == 0
            and first.relation_limit == 0
            and first.physical_predictions == 0
            and first.physical_races == 0
            and first.converter_control_frontier_sha256 is None
            and first.frontier_sha256 is None
            and first.frontier_work_sha256 is None
        )
        if not valid_stop:
            raise RunnerStop(
                STOP_PRESPEND, f"row {row_ordinal} corrupted an authentic static STOP"
            )
        return outcomes, f"row {row_ordinal} static-on stopped at {first.static_on_status}"

    converter_ready = (
        first.converter_control_status == engine.STOP_SOURCE_CERTIFICATE
        and first.converter_control_plan_build_attempts == 0
        and first.converter_build_attempts == 1
        and _valid_sha256(first.converter_control_frontier_sha256)
        and _valid_sha256(first.frontier_sha256)
    )
    if not converter_ready:
        raise RunnerStop(
            STOP_PRESPEND, f"row {row_ordinal} changed converter authority control"
        )
    if first.build_status != engine.READY:
        if (
            len(outcomes) != 1
            or first.selection_mode is not None
            or first.selection_status is not None
            or first.runtime_desired_reads != 0
            or first.exact_proof_count != 0
            or first.witness_count != 0
            or first.selected_proof_forest_sha256 is not None
        ):
            raise RunnerStop(
                STOP_PRESPEND, f"row {row_ordinal} corrupted a frontier STOP"
            )
        return outcomes, f"row {row_ordinal} frontier stopped at {first.build_status}"

    if len(outcomes) != 2:
        raise RunnerStop(
            STOP_PRESPEND, f"READY row {row_ordinal} omitted wrong/normal selections"
        )
    wrong, normal = outcomes
    if (
        (wrong.selection_mode, normal.selection_mode) != engine.SELECTION_ORDER
        or wrong.runtime_desired_reads != protocol.VISIBLE_CASE_COUNT
        or normal.runtime_desired_reads != protocol.VISIBLE_CASE_COUNT
        or wrong.exact_proof_count != 0
        or wrong.witness_count != 0
        or wrong.selected_proof_forest_sha256 is not None
        or wrong.relation_completed_units > normal.relation_completed_units
        or wrong.physical_predictions > normal.physical_predictions
        or wrong.physical_races > normal.physical_races
    ):
        raise RunnerStop(
            STOP_PRESPEND, f"READY row {row_ordinal} changed selection order/control"
        )
    selected = (
        normal.public_status == engine.SELECTED
        and normal.selection_status == engine.SELECTED
        and normal.exact_proof_count == 1
        and normal.witness_count == 1
        and _valid_sha256(normal.selected_proof_forest_sha256)
    )
    if selected:
        return outcomes, None
    if (
        normal.exact_proof_count != 0
        or normal.witness_count != 0
        or normal.selected_proof_forest_sha256 is not None
    ):
        raise RunnerStop(
            STOP_PRESPEND, f"normal row {row_ordinal} published a non-exact proof"
        )
    return outcomes, f"normal row {row_ordinal} did not publish exactly one proof"


def _ledger(
    row_outcomes: Sequence[Sequence[engine.RowTreatmentOutcome]],
) -> WorkLedger:
    finals = tuple(row[-1] for row in row_outcomes if row)
    selected_counts = tuple(item.source_selected_event_count for item in finals)
    source_queries = sum(item.source_query_count for item in finals)
    source_fitted = sum(item.source_fitted_glif_step_count for item in finals)
    static_off = sum(item.static_off_query_count for item in finals)
    static_on = sum(item.static_on_query_count for item in finals)
    static_queries = static_off + static_on
    static_dynamic_cap = sum(
        2 * _complete_static_queries(count) for count in selected_counts
    )
    static_fitted = sum(item.static_fitted_glif_step_count for item in finals)
    converter_builds = sum(item.converter_build_attempts for item in finals)
    plan_builds = sum(item.plan_build_attempts for item in finals)
    relation_completed = sum(item.relation_completed_units for item in finals)
    relation_attempted = sum(item.relation_attempted_units for item in finals)
    relation_limit = sum(item.relation_limit for item in finals)
    predictions = sum(item.physical_predictions for item in finals)
    races = sum(item.physical_races for item in finals)
    proofs = sum(item.witness_count for item in finals)
    fitted = source_fitted + static_fitted
    values = (
        source_queries,
        source_fitted,
        selected_counts,
        static_off,
        static_on,
        static_queries,
        static_dynamic_cap,
        static_fitted,
        fitted,
        converter_builds,
        plan_builds,
        relation_completed,
        relation_attempted,
        relation_limit,
        predictions,
        races,
        proofs,
    )
    return WorkLedger(*values, _digest("work-ledger", values))


def _work_stop(
    row_outcomes: Sequence[Sequence[engine.RowTreatmentOutcome]],
    *,
    require_complete: bool,
) -> str | None:
    finals = tuple(row[-1] for row in row_outcomes if row)
    work = _ledger(row_outcomes)
    for outcome in finals:
        if (
            outcome.relation_completed_units > outcome.relation_limit
            or outcome.relation_completed_units > engine.packed_converter.WORK_CAPS[0]
            or outcome.physical_predictions > engine.packed_converter.WORK_CAPS[1]
            or outcome.physical_races > engine.packed_converter.WORK_CAPS[2]
        ):
            return f"row {outcome.row_ordinal} exceeded packed plan work authority"
    if (
        work.source_query_count > protocol.WORK_FORMULA.source_query_cap
        or work.static_query_count > work.static_query_dynamic_cap
        or work.static_query_count > protocol.WORK_FORMULA.static_query_cap
        or work.fitted_glif_step_count > protocol.WORK_FORMULA.fitted_glif_step_cap
    ):
        return "aggregate V3 source/static work cap exceeded"
    if require_complete:
        expected_dynamic = protocol.static_query_count_for_selected_counts(
            work.selected_source_counts
        )
        if (
            len(finals) != len(protocol.ROWS)
            or work.source_query_count != protocol.WORK_FORMULA.source_query_cap
            or work.source_fitted_glif_step_count != work.source_query_count * 30
            or work.static_query_dynamic_cap != expected_dynamic
            or work.static_fitted_glif_step_count != work.static_query_count * 30
        ):
            return "complete V3 work differs from frozen geometry/dynamic-n formula"
    return None


def _execution(
    verdict: str,
    reason: str | None,
    prepared_sha256: str,
    row_outcomes: Sequence[Sequence[engine.RowTreatmentOutcome]],
) -> RunnerExecution:
    frozen = tuple(tuple(row) for row in row_outcomes)
    work = _ledger(frozen)
    values = (verdict, reason, prepared_sha256, frozen, work, 0)
    return RunnerExecution(*values, _digest("runner-execution", values))


def _execute_prepared(
    prepared: _PreparedRun,
    authority: _ExecutionAuthority,
    executor: _RowExecutor,
) -> RunnerExecution:
    """Bind each row once and continue after authentic semantic STOPs."""

    prepared_receipt = _lookup_prepared(prepared)
    _authenticate_authority(prepared, authority, executor)
    rows: list[tuple[engine.RowTreatmentOutcome, ...]] = []
    reasons: list[str] = []
    for row_ordinal, row in enumerate(protocol.ROWS):
        _authenticate_authority(prepared, authority, executor)
        supplied = executor.call(prepared_receipt.exact_preflight, row, row_ordinal)
        for outcome in supplied:
            executor.authenticate(outcome)
        outcomes, semantic_reason = _validate_row_outcomes(row_ordinal, supplied)
        rows.append(outcomes)
        if semantic_reason is not None:
            reasons.append(semantic_reason)
        work_reason = _work_stop(rows, require_complete=False)
        if work_reason is not None:
            raise RunnerStop(STOP_WORK_CAP, work_reason)

    work_reason = _work_stop(rows, require_complete=True)
    if work_reason is not None:
        raise RunnerStop(STOP_WORK_CAP, work_reason)
    unique_reasons = tuple(dict.fromkeys(reasons))
    if unique_reasons:
        return _execution(
            protocol.ONE_SHOT_STOP,
            "; ".join(unique_reasons),
            prepared.sha256,
            rows,
        )
    return _execution(protocol.ONE_SHOT_PASS, None, prepared.sha256, rows)


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
            "terminal_cohorts_sha256": PROTOCOL_TERMINAL_COHORTS_SHA256,
            "roster_sha256": PROTOCOL_ROSTER_SHA256,
            "execution_contract_sha256": PROTOCOL_EXECUTION_CONTRACT_SHA256,
            "work_formula_sha256": PROTOCOL_WORK_FORMULA_SHA256,
            "one_shot_order_sha256": PROTOCOL_ORDER_SHA256,
            "surface_boundary_sha256": PROTOCOL_SURFACE_SHA256,
        },
        "bound_engine": {
            "bank_commit": BOUND_ENGINE_BANK_COMMIT,
            "source_sha256": BOUND_ENGINE_SOURCE_SHA256,
        },
        "runner_source_sha256": _file_sha256(Path(__file__)),
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
            "state": "PUBLIC_TRAIN_JOINT_V3_ATTEMPT_SPENT",
            "exclusive_create": True,
            "flushed_and_fsynced_before_first_row": True,
        },
        "execution": execution,
        "claim_boundary": CLAIM_BOUNDARY,
        "scope": {
            "public_train_same_fit_finite_cue_replay": True,
            "dev_inputs": 0,
            "heldout_inputs": 0,
            "m0_inputs": 0,
            "c1_inputs": 0,
            "g0_inputs": 0,
            "protected_inputs": 0,
            "search_attempts": 0,
            "retries": 0,
            "cap_raises": 0,
            "row_substitutions": 0,
            "unseen_transfer_claim": False,
            "coding_capability_claim": False,
        },
    }


def _run_with_executor(executor: _RowExecutor) -> dict[str, object]:
    """Internal seam; fake executors require two noncanonical paths."""

    prepared = _prepare_pre_spend()
    authority = _create_spend_authority(prepared, executor)
    rows: tuple[tuple[engine.RowTreatmentOutcome, ...], ...] = ()
    try:
        execution = _execute_prepared(prepared, authority, executor)
    except Exception as error:
        execution = _execution(
            protocol.ONE_SHOT_STOP,
            f"{type(error).__name__}: {error}",
            prepared.sha256,
            rows,
        )
    artifact = _result_artifact(prepared, authority, execution)
    _write_json_exclusive(RESULT_PATH, artifact)
    return artifact


def run_one_shot() -> dict[str, object]:
    """Irreversibly spend and execute the exact fresh V3 roster once."""

    return _run_with_executor(_REAL_EXECUTOR)


def main() -> int:
    artifact = run_one_shot()
    print(artifact["verdict"])
    if artifact["stop_reason"]:
        print(artifact["stop_reason"])
    return 0 if artifact["verdict"] == protocol.ONE_SHOT_PASS else 1


CLAIM_BOUNDARY = (
    "prepared exclusive public-TRAIN same-FIT V3 runner only: exact zero-work "
    "preflight, durable no-retry spend, one bind per row, authentic STOP "
    "continuation, READY-only desired opening, and exact dynamic-n work receipts; "
    "no V3 execution/result/spend has occurred merely by importing or testing "
    "this source, and no unseen transfer, heldout, DEV, M0, C1, G0, protected, "
    "SWE-bench, or coding-capability evidence is claimed"
)


__all__ = [
    "CANONICAL_RESULT_PATH",
    "CANONICAL_SPENT_PATH",
    "CLAIM_BOUNDARY",
    "PRESPEND_CONTRACT_SHA256",
    "RESULT_PATH",
    "RUN_COMMAND",
    "SCHEMA",
    "SPENT_PATH",
    "STOP_ALREADY_SPENT",
    "STOP_PRESPEND",
    "STOP_WORK_CAP",
    "RunnerExecution",
    "RunnerStop",
    "WorkLedger",
    "run_one_shot",
]


if __name__ == "__main__":
    sys.exit(main())
