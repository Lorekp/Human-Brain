"""Irreversibly run the frozen Rust-documentation M0 causal gate once.

Chronology is owned here rather than delegated to the evaluator backend:

    committed READY_UNSPENT manifest
      -> exclusive, flushed, fsynced ATTEMPT_SPENT marker
      -> complete authenticated causal-control roster
      -> one selected materialization / executor call
      -> authenticated successful execution receipt
      -> first oracle-byte read
      -> immutable TERMINAL_PASS or TERMINAL_STOP result

Importing this module is inert.  Synthetic tests use a private noncanonical
backend seam and cannot target either canonical one-shot output path.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from experiments.r3_pilot import frozen_rust_docs_m0_causal_gate_v1_protocol as protocol

REPO_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_SPENT_PATH = protocol.DEFAULT_SPENT_PATH
CANONICAL_RESULT_PATH = protocol.DEFAULT_RESULT_PATH
CANONICAL_OWNER_LOCK_PATH = protocol.DEFAULT_OWNER_LOCK_PATH

SPEND_SCHEMA = "humanbrain.rust-docs-m0-causal-gate-spend.v1"
RESULT_SCHEMA = "humanbrain.rust-docs-m0-causal-gate-result.v1"
ATTEMPT_SPENT = "ATTEMPT_SPENT"
PASS = "PASS"
STOP = "STOP"

STOP_CONTROL = "STOP_CONTROL_INTEGRITY"
STOP_EXECUTION = "STOP_SELECTED_EXECUTION"
STOP_ORACLE = "STOP_ORACLE_AUTHENTICATION"
STOP_ORACLE_MISMATCH = "STOP_ORACLE_MISMATCH"
STOP_INTERRUPTED = "STOP_INTERRUPTED_SPEND"

RUN_CONFIRMATION = "RUN_FROZEN_RUST_DOCS_M0_CAUSAL_GATE_V1_ONCE"
SEAL_CONFIRMATION = "SEAL_FROZEN_RUST_DOCS_M0_CAUSAL_GATE_V1_INTERRUPTED_SPEND"
SHOW_CONFIRMATION = "SHOW_FROZEN_RUST_DOCS_M0_CAUSAL_GATE_V1_RESULT"

_SPEND_NAMESPACE = b"humanbrain-rust-docs-m0-causal-gate-spend-v1"
_RESULT_NAMESPACE = b"humanbrain-rust-docs-m0-causal-gate-result-v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")


class FrozenRustDocsM0RunnerError(RuntimeError):
    """The one-shot runner could not preserve its frozen chronology."""


class GateAlreadySpentError(FrozenRustDocsM0RunnerError):
    """The one-shot spend authority already exists."""


class GateAlreadyTerminalError(FrozenRustDocsM0RunnerError):
    """An immutable terminal result already exists."""


class GateAttemptActiveError(FrozenRustDocsM0RunnerError):
    """Another process still owns the crash-released one-shot authority."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_document(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _digest(namespace: bytes, value: object) -> str:
    return sha256(namespace + b"\0" + _canonical_json(value)).hexdigest()


def _plain(value: object) -> Any:
    return json.loads(_canonical_json(value))


def _valid_sha256(value: object) -> bool:
    return type(value) is str and _SHA256.fullmatch(value) is not None


def _file_sha256(path: Path) -> str:
    hasher = sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(block)
    except OSError as exc:
        raise FrozenRustDocsM0RunnerError(f"cannot authenticate {path}") from exc
    return hasher.hexdigest()


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _signed(namespace: bytes, payload: Mapping[str, object]) -> dict[str, object]:
    normalized = _plain(payload)
    return {**normalized, "sha256": _digest(namespace, normalized)}


def _publish_pending_exclusive(pending: Path, path: Path) -> None:
    """Atomically publish one complete same-directory file without replacement."""

    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        move_file = ctypes.WinDLL("kernel32", use_last_error=True).MoveFileExW
        move_file.argtypes = (wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD)
        move_file.restype = wintypes.BOOL
        if not move_file(str(pending), str(path), 0x00000008):  # MOVEFILE_WRITE_THROUGH
            error = ctypes.get_last_error()
            if error in (80, 183):  # ERROR_FILE_EXISTS / ERROR_ALREADY_EXISTS
                raise GateAlreadySpentError(f"refusing to overwrite {path}")
            raise OSError(error, ctypes.FormatError(error), str(path))
        return
    try:
        os.link(pending, path)
    except FileExistsError as exc:
        raise GateAlreadySpentError(f"refusing to overwrite {path}") from exc


def _write_json_exclusive(path: Path, value: Mapping[str, object]) -> str:
    """Publish complete bytes atomically without replacing an existing path."""

    if not path.parent.is_dir():
        raise FrozenRustDocsM0RunnerError(f"output directory is missing: {path.parent}")
    raw = _canonical_document(value)
    descriptor, pending_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".pending",
    )
    pending = Path(pending_name)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        if pending.read_bytes() != raw:
            raise FrozenRustDocsM0RunnerError("pending artifact bytes changed after fsync")
        _publish_pending_exclusive(pending, path)
        observed = path.read_bytes()
        if observed != raw:
            raise FrozenRustDocsM0RunnerError("published artifact bytes changed")
        if os.name != "nt":
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        return sha256(observed).hexdigest()
    finally:
        with suppress(FileNotFoundError):
            pending.unlink()


def _load_canonical(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenRustDocsM0RunnerError(f"invalid one-shot artifact: {path}") from exc
    if type(value) is not dict or raw != _canonical_document(value):
        raise FrozenRustDocsM0RunnerError(f"noncanonical one-shot artifact: {path}")
    return value, raw


@dataclass(frozen=True, slots=True)
class _RunPaths:
    spent: Path
    result: Path

    @property
    def owner_lock(self) -> Path:
        if _same_path(self.spent, CANONICAL_SPENT_PATH) and _same_path(
            self.result, CANONICAL_RESULT_PATH
        ):
            return CANONICAL_OWNER_LOCK_PATH
        return self.spent.with_name(f".{self.spent.name}.owner.lock")


class _AttemptOwnerLock:
    """One persistent sentinel lock released automatically when the process dies."""

    def __init__(self, handle: Any) -> None:
        self._handle = handle
        self._released = False

    @classmethod
    def acquire(cls, path: Path) -> _AttemptOwnerLock:
        if not path.parent.is_dir():
            raise FrozenRustDocsM0RunnerError(f"owner-lock directory is missing: {path.parent}")
        descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            if os.fstat(descriptor).st_size == 0:
                os.write(descriptor, b"\0")
                os.fsync(descriptor)
            handle = os.fdopen(descriptor, "r+b", buffering=0, closefd=True)
        except BaseException:
            os.close(descriptor)
            raise
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise GateAttemptActiveError("one-shot gate attempt is still active") from exc
        return cls(handle)

    def release(self) -> None:
        if self._released:
            return
        try:
            self._handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._released = True
            self._handle.close()

    def __enter__(self) -> _AttemptOwnerLock:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()


@dataclass(frozen=True, slots=True)
class _GateBackend:
    name: str
    real: bool
    prepare: Callable[[], object]
    prepared_record: Callable[[object], Mapping[str, object]]
    manifest_sha256: Callable[[object], str]
    run_controls: Callable[[object], object]
    controls_record: Callable[[object], Mapping[str, object]]
    execute_selected: Callable[[object, object], object]
    execution_record: Callable[[object], Mapping[str, object]]
    open_oracle: Callable[[object, object, object], Mapping[str, object]]


def _real_controls(prepared: object) -> object:
    if type(prepared) is not protocol._PreparedRustDocsM0GateV1:
        raise TypeError("exact prepared Rust gate required")
    return protocol._run_control_roster(prepared)


def _real_execute(prepared: object, controls: object) -> object:
    if type(prepared) is not protocol._PreparedRustDocsM0GateV1:
        raise TypeError("exact prepared Rust gate required")
    if type(controls) is not protocol.FrozenControlBundleV1:
        raise TypeError("exact frozen control bundle required")
    return protocol._execute_selected_once(prepared, controls)


def _real_oracle(
    prepared: object,
    controls: object,
    selected: object,
) -> Mapping[str, object]:
    if type(prepared) is not protocol._PreparedRustDocsM0GateV1:
        raise TypeError("exact prepared Rust gate required")
    if type(controls) is not protocol.FrozenControlBundleV1:
        raise TypeError("exact frozen control bundle required")
    if type(selected) is not protocol.AuthenticatedSelectedExecutionV1:
        raise TypeError("exact authenticated selected execution required")
    return protocol._load_oracle_after_authenticated_execution(
        prepared,
        controls,
        selected,
    )


_REAL_BACKEND = _GateBackend(
    name="frozen-rust-docs-m0-causal-gate-v1",
    real=True,
    prepare=protocol._prepare_ready_unspent,
    prepared_record=lambda value: value.ready_receipt.record(),
    manifest_sha256=lambda value: value.manifest_sha256,
    run_controls=_real_controls,
    controls_record=lambda value: value.record(),
    execute_selected=_real_execute,
    execution_record=lambda value: value.record(),
    open_oracle=_real_oracle,
)


def _same_path(first: Path, second: Path) -> bool:
    return first.resolve() == second.resolve()


def _authenticate_backend_paths(backend: _GateBackend, paths: _RunPaths) -> None:
    canonical_spent = _same_path(paths.spent, CANONICAL_SPENT_PATH)
    canonical_result = _same_path(paths.result, CANONICAL_RESULT_PATH)
    if backend is _REAL_BACKEND:
        if not backend.real or not (canonical_spent and canonical_result):
            raise FrozenRustDocsM0RunnerError(
                "real gate requires both exact canonical output paths"
            )
    elif backend.real or canonical_spent or canonical_result:
        raise FrozenRustDocsM0RunnerError(
            "synthetic backend may not target either canonical output path"
        )
    if _same_path(paths.spent, paths.result):
        raise FrozenRustDocsM0RunnerError("spend and result paths must be distinct")


def _assert_unspent(paths: _RunPaths) -> None:
    if paths.result.exists():
        raise GateAlreadyTerminalError(f"terminal result already exists: {paths.result}")
    if paths.spent.exists():
        raise GateAlreadySpentError(f"spend marker already exists: {paths.spent}")


def _expected_zero_work() -> dict[str, int]:
    return {
        "recall_count": 0,
        "selected_source_materialization_count": 0,
        "selected_executor_call_count": 0,
        "authenticated_rustc_compile_receipt_count": 0,
        "authenticated_binary_run_receipt_count": 0,
        "oracle_read_count": 0,
    }


def _matches_exact_int_mapping(value: object, expected: Mapping[str, int]) -> bool:
    return (
        type(value) is dict
        and set(value) == set(expected)
        and all(type(value[name]) is int and value[name] == expected[name] for name in expected)
    )


def _is_exact_splice_item(value: object) -> bool:
    return (
        type(value) is dict
        and set(value) == {"name", "kind", "status", "physical_recall_count"}
        and value.get("name") == "bound_memory_splice"
        and value.get("kind") == "authority_control"
        and value.get("status") == "STOP_BOUND_MEMORY_SPLICE"
        and type(value.get("physical_recall_count")) is int
        and value.get("physical_recall_count") == 0
    )


def _authenticate_prepared_record(record: Mapping[str, object]) -> dict[str, object]:
    copied = _plain(record)
    if type(copied) is not dict:
        raise FrozenRustDocsM0RunnerError("backend did not return READY_UNSPENT")
    try:
        receipt = protocol._ready_receipt_from_record(copied)
    except Exception as exc:
        raise FrozenRustDocsM0RunnerError(
            "backend did not return authenticated READY_UNSPENT"
        ) from exc
    if receipt.record() != copied:
        raise FrozenRustDocsM0RunnerError("READY_UNSPENT serialization changed")
    return copied


def _recall_from_record(record: object) -> protocol.knowledge_action.KnowledgeActionRecallV1:
    copied = _plain(record)
    if type(copied) is not dict:
        raise FrozenRustDocsM0RunnerError("serialized recall is not one exact record")
    fields = dict(copied)
    work_value = fields.get("work")
    if type(work_value) is not dict:
        raise FrozenRustDocsM0RunnerError("serialized recall work changed")
    try:
        fields["work"] = protocol.knowledge_action.KnowledgeActionRecallWorkV1(**work_value)
        fields["candidate_scores"] = tuple(tuple(value) for value in fields["candidate_scores"])
        fields["physical_raw_scores"] = tuple(tuple(row) for row in fields["physical_raw_scores"])
        if type(fields.get("physical_output_record")) is not dict:
            raise FrozenRustDocsM0RunnerError("serialized recall physical output changed")
        physical_output = dict(fields["physical_output_record"])
        for name in (
            "active_cells",
            "lane_spike_steps",
            "lane_simultaneous_spike_candidates",
        ):
            physical_output[name] = tuple(physical_output[name])
        fields["physical_output_record"] = physical_output
        recall = protocol.knowledge_action.KnowledgeActionRecallV1(**fields)
        recall.authenticate()
        expected_output = protocol.knowledge_action._CELLS.race_output(
            recall.physical_raw_scores
        ).record()
        if _canonical_json(recall.physical_output_record) != _canonical_json(expected_output):
            raise FrozenRustDocsM0RunnerError("serialized recall physical output changed")
    except Exception as exc:
        raise FrozenRustDocsM0RunnerError("serialized recall authentication failed") from exc
    if _plain(recall.record()) != copied:
        raise FrozenRustDocsM0RunnerError("serialized recall normalization changed")
    return recall


def _score_values(
    recall: protocol.knowledge_action.KnowledgeActionRecallV1,
) -> tuple[int, ...]:
    return tuple(score for _symbol, score in recall.candidate_scores)


def _authenticate_serialized_control_outcomes(
    prepared: Mapping[str, object],
    recalls: tuple[protocol.knowledge_action.KnowledgeActionRecallV1, ...],
    items: list[object],
) -> None:
    (
        primary,
        contrast,
        reversed_primary,
        cell_off,
        plasticity_off,
        shared,
        unknown,
        channel,
        mutex,
    ) = recalls
    candidate_set = prepared["candidate_set_sha256"]
    bound_checkpoint = prepared["bound_memory_checkpoint_sha256"]
    task_roots = prepared["task_projection_sha256s"]
    expected_task_sha256s = tuple(
        task_roots[name]
        for name in (
            "primary",
            "contrast",
            "primary",
            "primary",
            "primary",
            "shared_factor_thread",
            "unknown_factor",
            "single_factor_channel",
            "single_factor_mutex",
        )
    )
    if (
        tuple(recall.task_sha256 for recall in recalls) != expected_task_sha256s
        or any(recall.candidate_set_sha256 != candidate_set for recall in recalls)
        or any(
            recalls[index].checkpoint_sha256 != bound_checkpoint
            for index in (0, 1, 2, 3, 5, 6, 7, 8)
        )
        or plasticity_off.checkpoint_sha256 != prepared["no_plasticity_checkpoint_sha256"]
        or primary.status != protocol.knowledge_action.RECALLED
        or primary.selected_candidate_id != prepared["channel_action_handle"]
        or contrast.status != protocol.knowledge_action.RECALLED
        or contrast.selected_candidate_id != prepared["mutex_action_handle"]
        or reversed_primary.record() != primary.record()
        or cell_off.status != protocol.knowledge_action.STOP_CELL_OFF
        or plasticity_off.status != protocol.knowledge_action.STOP_PHYSICAL_RACE
        or any(_score_values(plasticity_off))
        or shared.status != protocol.knowledge_action.STOP_PHYSICAL_RACE
        or not _score_values(shared)
        or len(set(_score_values(shared))) != 1
        or _score_values(shared)[0] <= 0
        or unknown.status != protocol.knowledge_action.STOP_NO_MATCHED_FACTOR
        or any(_score_values(unknown))
        or channel.status != protocol.knowledge_action.RECALLED
        or channel.selected_candidate_id != prepared["channel_action_handle"]
        or mutex.status != protocol.knowledge_action.RECALLED
        or mutex.selected_candidate_id != prepared["mutex_action_handle"]
        or not _is_exact_splice_item(items[5])
    ):
        raise FrozenRustDocsM0RunnerError("serialized causal-control outcome changed")


def _authenticate_controls_record(
    record: Mapping[str, object],
    prepared: Mapping[str, object],
) -> dict[str, object]:
    copied = _plain(record)
    work = dict(_expected_zero_work())
    work["recall_count"] = 9
    items = copied.get("items")
    if (
        set(copied)
        != {
            "schema",
            "ready_sha256",
            "roster_sha256",
            "items",
            "control_count",
            "physical_recall_count",
            "work",
            "sha256",
        }
        or copied.get("schema") != protocol.CONTROL_SCHEMA
        or copied.get("ready_sha256") != prepared["sha256"]
        or copied.get("roster_sha256") != protocol.CONTROL_ROSTER_SHA256
        or type(copied.get("control_count")) is not int
        or copied.get("control_count") != len(protocol.CONTROL_ROSTER)
        or type(copied.get("physical_recall_count")) is not int
        or copied.get("physical_recall_count") != 9
        or not _matches_exact_int_mapping(copied.get("work"), work)
        or type(items) is not list
        or any(type(item) is not dict for item in items)
        or tuple(item["name"] for item in items) != protocol.CONTROL_ROSTER
        or not _valid_sha256(copied.get("sha256"))
        or copied.get("sha256")
        != protocol._digest(
            protocol._CONTROL_NAMESPACE,
            {key: value for key, value in copied.items() if key != "sha256"},
        )
    ):
        raise FrozenRustDocsM0RunnerError("causal-control bundle authentication failed")
    recall_item_indexes = (0, 1, 2, 3, 4, 6, 7, 8, 9)
    recalls: list[protocol.knowledge_action.KnowledgeActionRecallV1] = []
    for item_index in recall_item_indexes:
        item = items[item_index]
        if set(item) != {"name", "kind", "recall"} or item["kind"] != "physical_recall":
            raise FrozenRustDocsM0RunnerError("serialized control item changed")
        recall = _recall_from_record(item["recall"])
        if item["recall"] != _plain(recall.record()):
            raise FrozenRustDocsM0RunnerError("serialized control recall changed")
        recalls.append(recall)
    _authenticate_serialized_control_outcomes(
        prepared,
        tuple(recalls),
        items,
    )
    return copied


def _execution_from_record(
    record: object,
) -> protocol.rust_selected_action_executor.RustSelectedActionExecutionV1:
    copied = _plain(record)
    if type(copied) is not dict:
        raise FrozenRustDocsM0RunnerError("serialized executor receipt is not one record")
    expected_keys = {
        "materialization",
        "rustc_identity",
        "preparation",
        "dependency_receipt",
        "compile_receipt",
        "run_receipt",
        "outputs",
        "sha256",
    }
    if set(copied) != expected_keys:
        raise FrozenRustDocsM0RunnerError("serialized executor receipt schema changed")
    nested_names = expected_keys - {"outputs", "sha256"}
    if any(type(copied[name]) is not dict for name in nested_names):
        raise FrozenRustDocsM0RunnerError("serialized nested executor receipt changed")
    executor = protocol.rust_selected_action_executor
    try:
        materialization = protocol.rust_action_registry.RustActionMaterializationV1(
            **copied["materialization"]
        )
        identity_fields = dict(copied["rustc_identity"])
        for name in (
            "link_search_paths",
            "link_search_manifest_sha256s",
            "link_search_file_counts",
            "link_search_total_bytes",
        ):
            identity_fields[name] = tuple(identity_fields[name])
        identity = executor.RustcIdentityV1(**identity_fields)
        preparation_fields = dict(copied["preparation"])
        for name in ("compile_environment", "run_environment"):
            preparation_fields[name] = tuple(tuple(pair) for pair in preparation_fields[name])
        preparation = executor.RustExecutorPreparationV1(**preparation_fields)
        dependency_fields = dict(copied["dependency_receipt"])
        dependency_fields["dependency_paths"] = tuple(dependency_fields["dependency_paths"])
        dependency_fields["dependency_content_sha256s"] = tuple(
            tuple(pair) for pair in dependency_fields["dependency_content_sha256s"]
        )
        dependency = executor.RustDependencyReceiptV1(**dependency_fields)
        compile_fields = dict(copied["compile_receipt"])
        compile_fields["arguments"] = tuple(compile_fields["arguments"])
        compile_receipt = executor.RustCompileReceiptV1(**compile_fields)
        run_receipt = executor.RustRunReceiptV1(**copied["run_receipt"])
        execution = executor.RustSelectedActionExecutionV1(
            materialization=materialization,
            rustc_identity=identity,
            preparation=preparation,
            dependency_receipt=dependency,
            compile_receipt=compile_receipt,
            run_receipt=run_receipt,
            outputs=tuple(tuple(row) for row in copied["outputs"]),
            sha256=copied["sha256"],
        )
        execution.authenticate_structure()
    except Exception as exc:
        raise FrozenRustDocsM0RunnerError(
            "serialized executor receipt authentication failed"
        ) from exc
    if _plain(asdict(execution)) != copied:
        raise FrozenRustDocsM0RunnerError("serialized executor receipt normalization changed")
    return execution


def _authenticate_execution_record(
    record: Mapping[str, object],
    prepared: Mapping[str, object],
    controls: Mapping[str, object],
) -> dict[str, object]:
    copied = _plain(record)
    work = dict(_expected_zero_work())
    work.update(
        {
            "recall_count": 9,
            "selected_source_materialization_count": 1,
            "selected_executor_call_count": 1,
            "authenticated_rustc_compile_receipt_count": 1,
            "authenticated_binary_run_receipt_count": 1,
        }
    )
    if (
        type(copied) is not dict
        or set(copied) != {"schema", "control_bundle_sha256", "execution", "work", "sha256"}
        or copied.get("schema") != protocol.EXECUTION_SCHEMA
        or not _matches_exact_int_mapping(copied.get("work"), work)
        or copied.get("control_bundle_sha256") != controls["sha256"]
        or not _valid_sha256(copied.get("sha256"))
        or copied.get("sha256")
        != protocol._digest(
            protocol._EXECUTION_NAMESPACE,
            {key: value for key, value in copied.items() if key != "sha256"},
        )
    ):
        raise FrozenRustDocsM0RunnerError("selected execution wrapper changed")
    execution = _execution_from_record(copied["execution"])
    materialization = execution.materialization
    preparation = execution.preparation
    primary_recall_sha256 = controls["items"][0]["recall"]["sha256"]
    if (
        materialization.registry_sha256 != prepared["registry_sha256"]
        or materialization.recall_sha256 != primary_recall_sha256
        or materialization.action_handle != prepared["channel_action_handle"]
        or materialization.source != protocol.FROZEN_CHANNEL_SOURCE
        or materialization.source_sha256 != protocol._source_sha256(protocol.FROZEN_CHANNEL_SOURCE)
        or materialization.structural_sha256 != prepared["channel_structural_sha256"]
        or execution.rustc_identity.sha256 != prepared["rustc_identity_sha256"]
        or preparation.sha256 != prepared["executor_preparation_sha256"]
        or preparation.expected_registry_sha256 != prepared["registry_sha256"]
        or preparation.harness_sha256 != prepared["executor_harness_sha256"]
        or preparation.batch_protocol_sha256 != prepared["executor_batch_protocol_sha256"]
        or preparation.batch_sha256 != prepared["executor_batch_sha256"]
        or preparation.stdin_sha256 != prepared["executor_stdin_sha256"]
        or preparation.case_count != len(protocol.FROZEN_CASES)
    ):
        raise FrozenRustDocsM0RunnerError("selected execution receipt changed")
    return copied


def _authenticate_oracle_record(
    record: Mapping[str, object],
    execution: Mapping[str, object],
) -> dict[str, object]:
    copied = _plain(record)
    execution_receipt = _execution_from_record(execution["execution"])
    outputs_match = (
        protocol._digest(protocol._ORACLE_NAMESPACE, execution_receipt.outputs)
        == protocol.ORACLE_PAYLOAD_SHA256
    )
    work = dict(_expected_zero_work())
    work.update(
        {
            "recall_count": 9,
            "selected_source_materialization_count": 1,
            "selected_executor_call_count": 1,
            "authenticated_rustc_compile_receipt_count": 1,
            "authenticated_binary_run_receipt_count": 1,
            "oracle_read_count": 1,
        }
    )
    if (
        set(copied)
        != {
            "schema",
            "status",
            "oracle_file_sha256",
            "oracle_payload_sha256",
            "batch_payload_sha256",
            "execution_sha256",
            "outputs_match",
            "work",
            "sha256",
        }
        or copied.get("schema") != protocol.ADJUDICATION_SCHEMA
        or copied.get("status") not in (protocol.TERMINAL_PASS, protocol.TERMINAL_STOP)
        or type(copied.get("outputs_match")) is not bool
        or copied.get("outputs_match") is not outputs_match
        or not _matches_exact_int_mapping(copied.get("work"), work)
        or copied.get("oracle_file_sha256") != protocol.EXPECTED_ORACLE_FILE_SHA256
        or copied.get("oracle_payload_sha256") != protocol.ORACLE_PAYLOAD_SHA256
        or copied.get("batch_payload_sha256") != protocol.BATCH_PAYLOAD_SHA256
        or copied.get("execution_sha256") != execution["sha256"]
        or not _valid_sha256(copied.get("sha256"))
        or copied.get("status")
        != (protocol.TERMINAL_PASS if outputs_match else protocol.TERMINAL_STOP)
        or copied.get("sha256")
        != protocol._digest(
            protocol._ADJUDICATION_NAMESPACE,
            {key: value for key, value in copied.items() if key != "sha256"},
        )
    ):
        raise FrozenRustDocsM0RunnerError("oracle adjudication receipt changed")
    return copied


def _create_spend_marker(
    backend: _GateBackend,
    paths: _RunPaths,
    prepared_record: Mapping[str, object],
    manifest_sha256: str,
) -> tuple[dict[str, object], str]:
    _assert_unspent(paths)
    if not _valid_sha256(manifest_sha256):
        raise FrozenRustDocsM0RunnerError("prepared manifest root changed")
    payload = {
        "schema": SPEND_SCHEMA,
        "state": ATTEMPT_SPENT,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "backend": backend.name,
        "real_backend": backend.real,
        "manifest_sha256": manifest_sha256,
        "prepared": dict(prepared_record),
        "prepared_sha256": prepared_record["sha256"],
        "result_path": _display_path(paths.result),
        "no_retry": True,
        "exclusive_create": True,
        "flushed_and_fsynced_before_first_recall": True,
    }
    marker = _signed(_SPEND_NAMESPACE, payload)
    marker_file_sha256 = _write_json_exclusive(paths.spent, marker)
    if paths.result.exists():
        raise GateAlreadySpentError(
            "result appeared while spend marker was created; marker retained"
        )
    return marker, marker_file_sha256


def _load_spend_marker(
    path: Path,
    result_path: Path,
) -> tuple[dict[str, Any], str]:
    marker, raw = _load_canonical(path)
    payload = dict(marker)
    observed = payload.pop("sha256", None)
    if (
        set(marker)
        != {
            "schema",
            "state",
            "created_at_utc",
            "backend",
            "real_backend",
            "manifest_sha256",
            "prepared",
            "prepared_sha256",
            "result_path",
            "no_retry",
            "exclusive_create",
            "flushed_and_fsynced_before_first_recall",
            "sha256",
        }
        or marker.get("schema") != SPEND_SCHEMA
        or marker.get("state") != ATTEMPT_SPENT
        or type(marker.get("created_at_utc")) is not str
        or not marker["created_at_utc"]
        or type(marker.get("backend")) is not str
        or not marker["backend"]
        or type(marker.get("real_backend")) is not bool
        or marker.get("no_retry") is not True
        or marker.get("exclusive_create") is not True
        or marker.get("flushed_and_fsynced_before_first_recall") is not True
        or marker.get("result_path") != _display_path(result_path)
        or not _valid_sha256(observed)
        or observed != _digest(_SPEND_NAMESPACE, payload)
    ):
        raise FrozenRustDocsM0RunnerError("spend marker authentication failed")
    canonical_spent = _same_path(path, CANONICAL_SPENT_PATH)
    canonical_result = _same_path(result_path, CANONICAL_RESULT_PATH)
    if canonical_spent or canonical_result:
        if (
            not (canonical_spent and canonical_result)
            or marker["real_backend"] is not True
            or marker["backend"] != _REAL_BACKEND.name
        ):
            raise FrozenRustDocsM0RunnerError("canonical spend marker backend/path binding changed")
    elif marker["real_backend"] is not False:
        raise FrozenRustDocsM0RunnerError("noncanonical spend marker claims real backend authority")
    prepared = marker.get("prepared")
    if (
        type(prepared) is not dict
        or _authenticate_prepared_record(prepared) != prepared
        or marker.get("prepared_sha256") != prepared.get("sha256")
        or not _valid_sha256(marker.get("manifest_sha256"))
    ):
        raise FrozenRustDocsM0RunnerError("spend marker readiness binding changed")
    return marker, sha256(raw).hexdigest()


def _assert_spend_current(paths: _RunPaths, expected_file_sha256: str) -> None:
    _marker, observed = _load_spend_marker(paths.spent, paths.result)
    if observed != expected_file_sha256 or paths.result.exists():
        raise FrozenRustDocsM0RunnerError("one-shot spend authority changed")


def _terminal_artifact(
    *,
    verdict: str,
    stop_code: str | None,
    stop_reason: str | None,
    marker: Mapping[str, object],
    marker_file_sha256: str,
    prepared: Mapping[str, object],
    controls: Mapping[str, object] | None,
    execution: Mapping[str, object] | None,
    oracle: Mapping[str, object] | None,
    stage: str,
    control_phase_calls: int,
    selected_execution_calls: int,
    oracle_open_calls: int,
    interrupted_prior_work_unknown: bool,
) -> dict[str, object]:
    terminal_state = protocol.TERMINAL_PASS if verdict == PASS else protocol.TERMINAL_STOP
    payload = {
        "schema": RESULT_SCHEMA,
        "state": terminal_state,
        "verdict": verdict,
        "stop_code": stop_code,
        "stop_reason": stop_reason,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "claim_boundary": protocol.CLAIM_BOUNDARY,
        "manifest_sha256": marker["manifest_sha256"],
        "spend_marker_sha256": marker["sha256"],
        "spend_marker_file_sha256": marker_file_sha256,
        "prepared": dict(prepared),
        "controls": None if controls is None else dict(controls),
        "execution": None if execution is None else dict(execution),
        "oracle": None if oracle is None else dict(oracle),
        "last_completed_stage": stage,
        "runner_calls": {
            "control_phase_calls": control_phase_calls,
            "selected_execution_calls": selected_execution_calls,
            "oracle_open_calls": oracle_open_calls,
        },
        "interrupted_prior_work_unknown": interrupted_prior_work_unknown,
        "no_retry": True,
    }
    return _signed(_RESULT_NAMESPACE, payload)


def _write_terminal(paths: _RunPaths, artifact: Mapping[str, object]) -> dict[str, object]:
    _write_json_exclusive(paths.result, artifact)
    return _load_terminal_result(paths)


def _stop_code_for_stage(stage: str) -> str:
    if stage == "controls":
        return STOP_CONTROL
    if stage == "execution":
        return STOP_EXECUTION
    return STOP_ORACLE


def _run_with_backend_owned(
    backend: _GateBackend,
    paths: _RunPaths,
) -> dict[str, object]:
    """Run while the caller holds the persistent attempt-owner lock."""

    _assert_unspent(paths)

    # Everything above the durable marker is required to be zero-evaluation work.
    prepared_object = backend.prepare()
    prepared_record = _authenticate_prepared_record(backend.prepared_record(prepared_object))
    manifest_sha256 = backend.manifest_sha256(prepared_object)
    marker, marker_file_sha256 = _create_spend_marker(
        backend,
        paths,
        prepared_record,
        manifest_sha256,
    )

    controls_object: object | None = None
    controls_record: dict[str, object] | None = None
    execution_object: object | None = None
    execution_record: dict[str, object] | None = None
    oracle_record: dict[str, object] | None = None
    stage = "controls"
    control_calls = 0
    execution_calls = 0
    oracle_calls = 0
    try:
        _assert_spend_current(paths, marker_file_sha256)
        control_calls = 1
        controls_object = backend.run_controls(prepared_object)
        controls_record = _authenticate_controls_record(
            backend.controls_record(controls_object),
            prepared_record,
        )

        _assert_spend_current(paths, marker_file_sha256)
        stage = "execution"
        execution_calls = 1
        execution_object = backend.execute_selected(
            prepared_object,
            controls_object,
        )
        execution_record = _authenticate_execution_record(
            backend.execution_record(execution_object),
            prepared_record,
            controls_record,
        )

        _assert_spend_current(paths, marker_file_sha256)
        stage = "oracle"
        oracle_calls = 1
        oracle_record = _authenticate_oracle_record(
            backend.open_oracle(
                prepared_object,
                controls_object,
                execution_object,
            ),
            execution_record,
        )
    except Exception as exc:
        last_completed = {
            "controls": "spend_marker",
            "execution": "controls",
            "oracle": "execution",
        }[stage]
        artifact = _terminal_artifact(
            verdict=STOP,
            stop_code=_stop_code_for_stage(stage),
            stop_reason=f"{type(exc).__name__}: {stage} phase stopped",
            marker=marker,
            marker_file_sha256=marker_file_sha256,
            prepared=prepared_record,
            controls=controls_record,
            execution=execution_record,
            oracle=oracle_record,
            stage=last_completed,
            control_phase_calls=control_calls,
            selected_execution_calls=execution_calls,
            oracle_open_calls=oracle_calls,
            interrupted_prior_work_unknown=False,
        )
        return _write_terminal(paths, artifact)

    if oracle_record["outputs_match"] is not True:
        verdict = STOP
        stop_code = STOP_ORACLE_MISMATCH
        stop_reason = "authenticated selected output differed from frozen oracle"
    else:
        verdict = PASS
        stop_code = None
        stop_reason = None
    artifact = _terminal_artifact(
        verdict=verdict,
        stop_code=stop_code,
        stop_reason=stop_reason,
        marker=marker,
        marker_file_sha256=marker_file_sha256,
        prepared=prepared_record,
        controls=controls_record,
        execution=execution_record,
        oracle=oracle_record,
        stage="oracle_adjudicated",
        control_phase_calls=control_calls,
        selected_execution_calls=execution_calls,
        oracle_open_calls=oracle_calls,
        interrupted_prior_work_unknown=False,
    )
    return _write_terminal(paths, artifact)


def _run_with_backend(backend: _GateBackend, paths: _RunPaths) -> dict[str, object]:
    """Private synthetic seam.  Path authority is checked before preparation."""

    _authenticate_backend_paths(backend, paths)
    with _AttemptOwnerLock.acquire(paths.owner_lock):
        return _run_with_backend_owned(backend, paths)


def run_one_shot() -> dict[str, object]:
    """Irreversibly spend and execute the exact canonical gate once."""

    return _run_with_backend(
        _REAL_BACKEND,
        _RunPaths(CANONICAL_SPENT_PATH, CANONICAL_RESULT_PATH),
    )


def _authenticate_canonical_manifest_binding(
    paths: _RunPaths,
    marker: Mapping[str, object],
) -> None:
    """Anchor canonical terminal state to the committed static READY manifest."""

    if not (
        _same_path(paths.spent, CANONICAL_SPENT_PATH)
        and _same_path(paths.result, CANONICAL_RESULT_PATH)
    ):
        return
    try:
        manifest, raw = protocol._load_manifest()
        protocol._authenticate_manifest_static(manifest, raw)
        readiness = protocol._ready_receipt_from_record(manifest["readiness"]).record()
    except Exception as exc:
        raise FrozenRustDocsM0RunnerError(
            "canonical READY_UNSPENT manifest authentication failed"
        ) from exc
    if (
        marker.get("manifest_sha256") != manifest["manifest_sha256"]
        or marker.get("prepared") != readiness
    ):
        raise FrozenRustDocsM0RunnerError("canonical spend manifest binding changed")


def _load_terminal_result(paths: _RunPaths) -> dict[str, object]:
    result, raw = _load_canonical(paths.result)
    payload = dict(result)
    observed = payload.pop("sha256", None)
    if (
        set(result)
        != {
            "schema",
            "state",
            "verdict",
            "stop_code",
            "stop_reason",
            "created_at_utc",
            "claim_boundary",
            "manifest_sha256",
            "spend_marker_sha256",
            "spend_marker_file_sha256",
            "prepared",
            "controls",
            "execution",
            "oracle",
            "last_completed_stage",
            "runner_calls",
            "interrupted_prior_work_unknown",
            "no_retry",
            "sha256",
        }
        or result.get("schema") != RESULT_SCHEMA
        or result.get("state") not in (protocol.TERMINAL_PASS, protocol.TERMINAL_STOP)
        or result.get("verdict") not in (PASS, STOP)
        or result.get("claim_boundary") != protocol.CLAIM_BOUNDARY
        or type(result.get("created_at_utc")) is not str
        or not result["created_at_utc"]
        or not _valid_sha256(result.get("manifest_sha256"))
        or not _valid_sha256(result.get("spend_marker_sha256"))
        or not _valid_sha256(result.get("spend_marker_file_sha256"))
        or result.get("no_retry") is not True
        or not _valid_sha256(observed)
        or observed != _digest(_RESULT_NAMESPACE, payload)
    ):
        raise FrozenRustDocsM0RunnerError("terminal result authentication failed")
    marker, marker_file_sha256 = _load_spend_marker(paths.spent, paths.result)
    _authenticate_canonical_manifest_binding(paths, marker)
    if (
        result.get("spend_marker_sha256") != marker["sha256"]
        or result.get("spend_marker_file_sha256") != marker_file_sha256
        or result.get("prepared") != marker["prepared"]
        or result.get("manifest_sha256") != marker["manifest_sha256"]
        or sha256(raw).hexdigest() != _file_sha256(paths.result)
    ):
        raise FrozenRustDocsM0RunnerError("terminal result spend binding changed")
    controls = result.get("controls")
    execution = result.get("execution")
    oracle = result.get("oracle")
    if controls is not None:
        if type(controls) is not dict:
            raise FrozenRustDocsM0RunnerError("terminal controls record changed")
        _authenticate_controls_record(controls, result["prepared"])
    if execution is not None:
        if type(execution) is not dict or controls is None:
            raise FrozenRustDocsM0RunnerError("terminal execution record changed")
        _authenticate_execution_record(execution, result["prepared"], controls)
    if oracle is not None:
        if type(oracle) is not dict or execution is None:
            raise FrozenRustDocsM0RunnerError("terminal oracle record changed")
        _authenticate_oracle_record(oracle, execution)
    if controls is not None and controls.get("ready_sha256") != result["prepared"].get("sha256"):
        raise FrozenRustDocsM0RunnerError("terminal control readiness root changed")
    if execution is not None and execution.get("control_bundle_sha256") != controls.get("sha256"):
        raise FrozenRustDocsM0RunnerError("terminal execution control root changed")
    if oracle is not None and oracle.get("execution_sha256") != execution.get("sha256"):
        raise FrozenRustDocsM0RunnerError("terminal oracle execution root changed")

    calls = result.get("runner_calls")
    all_calls = {
        "control_phase_calls": 1,
        "selected_execution_calls": 1,
        "oracle_open_calls": 1,
    }
    if result["verdict"] == PASS:
        valid_semantics = (
            result["state"] == protocol.TERMINAL_PASS
            and result.get("stop_code") is None
            and result.get("stop_reason") is None
            and controls is not None
            and execution is not None
            and oracle is not None
            and oracle.get("outputs_match") is True
            and _matches_exact_int_mapping(calls, all_calls)
            and result.get("last_completed_stage") == "oracle_adjudicated"
            and result.get("interrupted_prior_work_unknown") is False
        )
    else:
        stop_code = result.get("stop_code")
        shapes = {
            STOP_CONTROL: (
                None,
                None,
                None,
                {**all_calls, "selected_execution_calls": 0, "oracle_open_calls": 0},
                "spend_marker",
                False,
            ),
            STOP_EXECUTION: (
                "present",
                None,
                None,
                {**all_calls, "oracle_open_calls": 0},
                "controls",
                False,
            ),
            STOP_ORACLE: (
                "present",
                "present",
                None,
                all_calls,
                "execution",
                False,
            ),
            STOP_ORACLE_MISMATCH: (
                "present",
                "present",
                "present",
                all_calls,
                "oracle_adjudicated",
                False,
            ),
            STOP_INTERRUPTED: (
                None,
                None,
                None,
                {
                    "control_phase_calls": 0,
                    "selected_execution_calls": 0,
                    "oracle_open_calls": 0,
                },
                "interrupted_spend_sealed",
                True,
            ),
        }
        shape = shapes.get(stop_code)
        valid_semantics = (
            result["state"] == protocol.TERMINAL_STOP
            and type(result.get("stop_reason")) is str
            and bool(result["stop_reason"])
            and shape is not None
            and (controls is not None) == (shape[0] == "present")
            and (execution is not None) == (shape[1] == "present")
            and (oracle is not None) == (shape[2] == "present")
            and _matches_exact_int_mapping(calls, shape[3])
            and result.get("last_completed_stage") == shape[4]
            and result.get("interrupted_prior_work_unknown") is shape[5]
            and (stop_code != STOP_ORACLE_MISMATCH or oracle.get("outputs_match") is False)
        )
    if not valid_semantics:
        raise FrozenRustDocsM0RunnerError("terminal result semantics changed")
    return result


def load_terminal_result() -> dict[str, object]:
    """Load the immutable canonical PASS/STOP artifact without preparing the gate."""

    paths = _RunPaths(CANONICAL_SPENT_PATH, CANONICAL_RESULT_PATH)
    with _AttemptOwnerLock.acquire(paths.owner_lock):
        return _load_terminal_result(paths)


def _seal_interrupted_spend_owned(paths: _RunPaths) -> dict[str, object]:
    """Seal a marker while the caller holds the persistent owner lock."""

    if paths.result.exists():
        raise GateAlreadyTerminalError(f"terminal result already exists: {paths.result}")
    marker, marker_file_sha256 = _load_spend_marker(paths.spent, paths.result)
    _authenticate_canonical_manifest_binding(paths, marker)
    prepared = marker.get("prepared")
    if type(prepared) is not dict:
        raise FrozenRustDocsM0RunnerError("interrupted marker lost prepared receipt")
    artifact = _terminal_artifact(
        verdict=STOP,
        stop_code=STOP_INTERRUPTED,
        stop_reason=(
            "durable spend existed without a result; prior attempt work is unknown "
            "and no retry is authorized"
        ),
        marker=marker,
        marker_file_sha256=marker_file_sha256,
        prepared=prepared,
        controls=None,
        execution=None,
        oracle=None,
        stage="interrupted_spend_sealed",
        control_phase_calls=0,
        selected_execution_calls=0,
        oracle_open_calls=0,
        interrupted_prior_work_unknown=True,
    )
    return _write_terminal(paths, artifact)


def _seal_interrupted_spend(paths: _RunPaths) -> dict[str, object]:
    with _AttemptOwnerLock.acquire(paths.owner_lock):
        return _seal_interrupted_spend_owned(paths)


def seal_interrupted_spend_as_stop() -> dict[str, object]:
    """Seal a marker-only canonical crash as immutable STOP without evaluation."""

    return _seal_interrupted_spend(_RunPaths(CANONICAL_SPENT_PATH, CANONICAL_RESULT_PATH))


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments == [RUN_CONFIRMATION]:
        artifact = run_one_shot()
    elif arguments == [SEAL_CONFIRMATION]:
        artifact = seal_interrupted_spend_as_stop()
    elif arguments == [SHOW_CONFIRMATION]:
        artifact = load_terminal_result()
    else:
        print(
            "provide exactly one explicit RUN, SEAL, or SHOW confirmation token",
            file=sys.stderr,
        )
        return 2
    print(artifact["verdict"])
    if artifact.get("stop_code"):
        print(artifact["stop_code"])
    return 0 if artifact["verdict"] == PASS else 1


__all__ = [
    "ATTEMPT_SPENT",
    "CANONICAL_RESULT_PATH",
    "CANONICAL_SPENT_PATH",
    "PASS",
    "RESULT_SCHEMA",
    "RUN_CONFIRMATION",
    "SEAL_CONFIRMATION",
    "SHOW_CONFIRMATION",
    "SPEND_SCHEMA",
    "STOP",
    "FrozenRustDocsM0RunnerError",
    "GateAlreadySpentError",
    "GateAlreadyTerminalError",
    "load_terminal_result",
    "run_one_shot",
    "seal_interrupted_spend_as_stop",
]


if __name__ == "__main__":
    sys.exit(main())
