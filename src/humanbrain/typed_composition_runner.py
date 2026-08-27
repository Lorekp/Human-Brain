"""Plan, seal, score, and evidence the frozen typed-composition protocol."""

from __future__ import annotations

import base64
import ctypes
import json
import math
import os
import platform
import re
import stat
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np

import humanbrain as humanbrain_package
from humanbrain import cells as cells_runtime
from humanbrain import cuda as cuda_runtime
from humanbrain import human_cell_identity as human_cell_identity_runtime
from humanbrain import native as native_runtime
from humanbrain import readout as readout_runtime
from humanbrain import recurrent as recurrent_runtime
from humanbrain import recurrent_state as recurrent_state_runtime
from humanbrain import topology as topology_runtime
from humanbrain import typed_composition as typed_composition_runtime
from humanbrain import typed_program
from humanbrain import typed_program_types as typed_program_types_runtime
from humanbrain.typed_composition import (
    CompositionProtocol,
    CompositionTask,
    LaneCommitment,
    load_composition_protocol,
)
from humanbrain.typed_program import (
    Completion,
    FrozenCheckpoint,
    Layer1Protocol,
    Program,
    ProgramRaceAudit,
    TypedSignature,
    compile,
    complete,
    grow,
)


class ProtocolConfirmationError(RuntimeError):
    """Execution was requested without the exact frozen raw protocol hash."""


class EvidenceExistsError(RuntimeError):
    """An exclusive evidence path already exists."""


_MAX_SOURCE_FILE_BYTES = 1024 * 1024
# Recursive native build inputs include the audited target-specific Windows
# bindings even when a plan is produced on another host.
_MAX_SOURCE_BUNDLE_BYTES = 32 * 1024 * 1024
_MOVEFILE_WRITE_THROUGH = 0x00000008
_WINDOWS_ALREADY_EXISTS = frozenset({80, 183})
_PROTOCOL_RELATIVE_PATH = Path("benchmarks/typed_composition_protocol_v1.json")
_PRODUCTION_ATTEMPT_RELATIVE_PATH = Path(
    "artifacts/typed_composition_a012_v1"
)
_PRODUCTION_CLAIM_MARKER_BYTES = (
    b'{"authority":"typed-composition-a012-v1-exclusive-protocol-cohort-claim",'
    b'"rerun_attempt_allowed":false}'
)
_PRODUCTION_INVALID_MARKER_BYTES = (
    b'{"authority":"typed-composition-a012-v1-invalid-terminal-outcome",'
    b'"terminal_authority_invalid":true}'
)
_PRODUCTION_COMPLETE_MARKER_BYTES = (
    b'{"authority":"typed-composition-a012-v1-terminal-envelope-complete"}'
)
_LOCAL_RUNTIME_MODULES = (
    sys.modules[__name__],
    humanbrain_package,
    cells_runtime,
    cuda_runtime,
    human_cell_identity_runtime,
    native_runtime,
    readout_runtime,
    recurrent_runtime,
    recurrent_state_runtime,
    topology_runtime,
    typed_composition_runtime,
    typed_program,
    typed_program_types_runtime,
)


@dataclass(frozen=True, slots=True)
class SealedCompletion:
    sequence_number: int
    lane: str
    cohort: str
    task_id: str
    record_bytes: bytes
    sha256: str


@dataclass(frozen=True, slots=True)
class A2CapabilityMetrics:
    train_replay_tasks: int
    train_cases: int
    calibration_replay_tasks: int
    calibration_cases: int
    common_integrity_gates_passed: bool
    decision_collision_count: int
    competitive_train_margin_count: int
    minimum_train_target_margin_pa: float

    def __post_init__(self) -> None:
        integer_names = (
            "train_replay_tasks",
            "train_cases",
            "calibration_replay_tasks",
            "calibration_cases",
            "decision_collision_count",
            "competitive_train_margin_count",
        )
        if any(type(getattr(self, name)) is not int for name in integer_names):
            raise TypeError("A2 capability counts must be exact integers")
        if type(self.common_integrity_gates_passed) is not bool:
            raise TypeError("A2 common-integrity result must be an exact boolean")
        if type(self.minimum_train_target_margin_pa) is not float:
            raise TypeError("A2 minimum target margin must be an exact float")


@dataclass(frozen=True, slots=True)
class ScoredDiagnostics:
    record: dict[str, object]
    a2_metrics: A2CapabilityMetrics
    seal_integrity_passed: bool
    race_integrity_passed: bool
    checkpoint_cell_synapse_passed: bool
    checkpoint_maps_passed: bool


@dataclass(frozen=True, slots=True)
class ScoredEvaluation:
    record: dict[str, object]
    terminal_summary: dict[str, object]
    a2_metrics: A2CapabilityMetrics
    a2_eligible: bool


@dataclass(frozen=True, slots=True)
class _ProductionExecutionGates:
    protocol_confirmation_passed: bool
    protocol_unchanged: bool
    source_bundle_confirmation_passed: bool
    source_bundle_unchanged: bool
    fixed_attempt_claim_passed: bool
    native_backend_passed: bool
    native_library_identity_passed: bool
    checkpoint_raw_bytes_passed: bool
    checkpoint_identity_passed: bool
    checkpoint_cell_synapse_passed: bool
    checkpoint_accounting_passed: bool
    observation_strengthen_accounting_passed: bool
    completion_attempt_accounting_passed: bool
    race_integrity_passed: bool
    seal_integrity_passed: bool
    preterminal_evidence_passed: bool
    terminal_failure_absent: bool

    def __post_init__(self) -> None:
        if any(
            type(getattr(self, name)) is not bool
            for name in self.__dataclass_fields__
        ):
            raise TypeError("production execution gates must be exact booleans")

    def to_record(self) -> dict[str, bool]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }

    def all_passed(self) -> bool:
        return all(self.to_record().values())


@dataclass(frozen=True, slots=True)
class _ExecutionClock:
    wall_ns: Callable[[], int]
    process_ns: Callable[[], int]


@dataclass(frozen=True, slots=True)
class _ExecutionRun:
    root: Path
    protocol_path: Path
    protocol: CompositionProtocol
    plan: dict[str, object]
    runtime_backend: dict[str, object]
    runtime_provenance: dict[str, object]
    run_directory: Path
    execution_plan_bytes: bytes
    confirmed_protocol_sha256: str | None
    confirmed_source_bundle_sha256: str | None


@dataclass(frozen=True, slots=True)
class _ProductionExecutionRun(_ExecutionRun):
    claimed_directory_identity: tuple[int, int]


@dataclass(frozen=True, slots=True)
class _SyntheticExecutionRun(_ExecutionRun):
    pass


@dataclass(frozen=True, slots=True)
class _TrainedLanes:
    checkpoint_bytes: dict[str, bytes]
    checkpoints: dict[str, object]
    checkpoint_hashes: dict[str, str]
    checkpoint_cell_synapse_hashes: dict[str, str]
    checkpoint_accounting: dict[str, dict[str, int]]
    performance_lanes: dict[str, dict[str, int]]
    native_library_identity_before: dict[str, int | str] | None


_SYSTEM_CLOCK = _ExecutionClock(
    wall_ns=time.perf_counter_ns,
    process_ns=time.process_time_ns,
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _resolved_source(root: Path, relative_path: str) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("source bundle path must be non-empty text")
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("source bundle paths must be repository relative")
    resolved_root = root.resolve()
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError("source bundle path escapes the repository") from error
    if not resolved.is_file():
        raise ValueError(f"source bundle path does not exist: {relative_path}")
    return resolved


def _required_local_source_paths(root: Path) -> tuple[str, ...]:
    """Return imported local Python sources relative to their one checkout."""
    resolved_root = root.resolve()
    paths: list[str] = []
    for module in _LOCAL_RUNTIME_MODULES:
        module_path_value = getattr(module, "__file__", None)
        if not isinstance(module_path_value, str):
            raise RuntimeError("typed-composition runtime module has no source path")
        module_path = Path(module_path_value).resolve()
        try:
            relative = module_path.relative_to(resolved_root)
        except ValueError as error:
            raise RuntimeError(
                "typed-composition runtime modules come from different checkouts"
            ) from error
        if module_path.suffix != ".py" or not module_path.is_file():
            raise RuntimeError("typed-composition runtime module is not a Python source file")
        paths.append(relative.as_posix())
    if len(paths) != len(set(paths)):
        raise RuntimeError("typed-composition runtime source registry contains duplicates")
    return tuple(sorted(paths))


def _production_repository_root() -> Path:
    package_path = Path(humanbrain_package.__file__).resolve()
    root = package_path.parents[2]
    if package_path != root / "src" / "humanbrain" / "__init__.py":
        raise RuntimeError("imported humanbrain package has an unexpected repository layout")
    _required_local_source_paths(root)
    return root


def _require_production_checkout(root: Path, protocol_path: Path) -> None:
    imported_root = _production_repository_root()
    expected_protocol = (imported_root / _PROTOCOL_RELATIVE_PATH).resolve()
    if root.resolve() != imported_root or protocol_path.resolve() != expected_protocol:
        raise ValueError("production execution must use the imported repository checkout")


def _lexical_absolute(path: Path) -> Path:
    """Normalize an absolute path without following a filesystem link."""
    if not isinstance(path, Path):
        raise TypeError("filesystem path must be a Path")
    return Path(os.path.abspath(os.fspath(path)))


def _path_overlaps_physical_namespace(candidate: Path, protected: Path) -> bool:
    """Detect lexical and existing-prefix aliases into a protected namespace."""
    candidate = _lexical_absolute(candidate)
    protected = _lexical_absolute(protected)
    if os.name == "nt" and candidate.anchor.startswith("\\\\"):
        # Synthetic evidence has no need for UNC/device spellings. Refusing
        # them avoids alternate-volume aliases into the fixed local claim.
        return True

    def component_key(path: Path) -> tuple[str, ...]:
        if os.name == "nt":
            return tuple(part.rstrip(" .").casefold() for part in path.parts)
        return tuple(path.parts)

    candidate_key = component_key(candidate)
    protected_key = component_key(protected)
    if candidate_key[: len(protected_key)] == protected_key:
        return True
    if candidate == protected or protected in candidate.parents:
        return True
    try:
        resolved_candidate = candidate.resolve(strict=False)
        resolved_protected = protected.resolve(strict=False)
    except OSError:
        return True
    if (
        resolved_candidate == resolved_protected
        or resolved_protected in resolved_candidate.parents
    ):
        return True

    def existing_prefixes(path: Path) -> tuple[tuple[Path, tuple[str, ...]], ...]:
        prefixes: list[tuple[Path, tuple[str, ...]]] = []
        suffix: list[str] = []
        cursor = path
        while True:
            if os.path.lexists(cursor):
                normalized_suffix = tuple(
                    part.rstrip(" .").casefold() if os.name == "nt" else part
                    for part in reversed(suffix)
                )
                prefixes.append((cursor, normalized_suffix))
            parent = cursor.parent
            if parent == cursor:
                return tuple(prefixes)
            suffix.append(cursor.name)
            cursor = parent

    candidate_prefixes = existing_prefixes(candidate)
    protected_prefixes = existing_prefixes(protected)
    for candidate_prefix, candidate_suffix in candidate_prefixes:
        for protected_prefix, protected_suffix in protected_prefixes:
            try:
                same_prefix = os.path.samefile(candidate_prefix, protected_prefix)
            except OSError:
                return True
            if same_prefix and candidate_suffix[: len(protected_suffix)] == (
                protected_suffix
            ):
                return True
    return False


def _is_link_or_reparse(path: Path) -> bool:
    metadata = os.lstat(path)
    attributes = int(getattr(metadata, "st_file_attributes", 0))
    return stat.S_ISLNK(metadata.st_mode) or bool(
        attributes & int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    )


def _is_plain_directory(path: Path) -> bool:
    try:
        metadata = os.lstat(_lexical_absolute(path))
    except OSError:
        return False
    attributes = int(getattr(metadata, "st_file_attributes", 0))
    return (
        stat.S_ISDIR(metadata.st_mode)
        and not stat.S_ISLNK(metadata.st_mode)
        and not bool(
            attributes & int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
        )
    )


def _require_plain_directory(path: Path) -> Path:
    """Require one existing real directory, never a symlink or reparse point."""
    absolute = _lexical_absolute(path)
    try:
        metadata = os.lstat(absolute)
    except OSError as error:
        raise OSError(f"required durable directory does not exist: {absolute}") from error
    if _is_link_or_reparse(absolute):
        raise OSError(
            f"durable namespace contains a symbolic link or reparse point: {absolute}"
        )
    if not stat.S_ISDIR(metadata.st_mode):
        raise NotADirectoryError(f"durable namespace component is not a directory: {absolute}")
    return absolute


def _plain_directory_identity(path: Path) -> tuple[int, int]:
    """Return the stable filesystem identity of one real directory."""
    absolute = _require_plain_directory(path)
    metadata = os.lstat(absolute)
    if _is_link_or_reparse(absolute) or not stat.S_ISDIR(metadata.st_mode):
        raise OSError(f"durable directory identity changed during inspection: {absolute}")
    return (int(metadata.st_dev), int(metadata.st_ino))


def _production_attempt_directory(root: Path) -> Path:
    """Return the one fixed physical attempt claim for this protocol/cohort."""
    absolute_root = _require_plain_directory(root)
    artifacts = _require_plain_directory(absolute_root / "artifacts")
    attempt = _lexical_absolute(absolute_root / _PRODUCTION_ATTEMPT_RELATIVE_PATH)
    if attempt.parent != artifacts:
        raise RuntimeError("production attempt path is not below the fixed artifacts directory")
    return attempt


def _production_attempt_staging_directory(root: Path) -> Path:
    """Return the deterministic Windows indeterminate-claim namespace."""
    attempt = _production_attempt_directory(root)
    return attempt.with_name(f".{attempt.name}.claiming")


def _production_attempt_claim_marker(root: Path) -> Path:
    """Return the sole direct-create durable claim marker."""
    attempt = _production_attempt_directory(root)
    return attempt.with_name(f".{attempt.name}.claimed")


def _production_attempt_invalid_marker(root: Path) -> Path:
    attempt = _production_attempt_directory(root)
    return attempt.with_name(f".{attempt.name}.invalid")


def _production_attempt_complete_marker(root: Path) -> Path:
    attempt = _production_attempt_directory(root)
    return attempt.with_name(f".{attempt.name}.complete")


def _create_durable_exclusive_marker(
    path: Path,
    value: bytes,
    *,
    on_created: Callable[[], None] | None = None,
) -> None:
    """Create one fixed marker directly and durably, with no temp-name gap."""
    if type(value) is not bytes:
        raise TypeError("durable marker value must be exact bytes")
    path = _lexical_absolute(path)
    _require_plain_directory(path.parent)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    try:
        file_descriptor = os.open(path, flags, 0o600)
    except FileExistsError as error:
        raise EvidenceExistsError(
            "fixed typed-composition protocol/cohort attempt already exists"
        ) from error
    try:
        if on_created is not None:
            on_created()
        _write_temp_fully(file_descriptor, value)
    except BaseException:
        with suppress(OSError):
            os.close(file_descriptor)
        raise
    os.close(file_descriptor)
    if os.name != "nt":
        _sync_posix_directory(path.parent)
    _verify_exact_record(path, value)


def _create_durable_claim_marker(
    path: Path,
    *,
    on_created: Callable[[], None] | None = None,
) -> None:
    """Create the fixed claim directly and flush it before any later side effect."""
    _create_durable_exclusive_marker(
        path,
        _PRODUCTION_CLAIM_MARKER_BYTES,
        on_created=on_created,
    )


def _durably_invalidate_production_attempt(root: Path) -> None:
    """Poison every post-claim failure outside the possibly unwritable run dir."""
    marker = _production_attempt_invalid_marker(root)
    if os.path.lexists(marker):
        return
    with suppress(EvidenceExistsError):
        _create_durable_exclusive_marker(marker, _PRODUCTION_INVALID_MARKER_BYTES)
    if not os.path.lexists(marker):
        raise RuntimeError("production invalid marker was not durably published")


def _durably_complete_production_attempt(root: Path) -> None:
    """Commit terminal authority only after the summary is durably visible."""
    if os.path.lexists(_production_attempt_invalid_marker(root)):
        raise RuntimeError("invalid production attempt cannot be completed")
    _create_durable_exclusive_marker(
        _production_attempt_complete_marker(root),
        _PRODUCTION_COMPLETE_MARKER_BYTES,
    )


def _bounded_source_bytes(source: Path, aggregate_remaining: int) -> bytes:
    if aggregate_remaining <= 0:
        raise ValueError("source bundle exceeds the aggregate byte limit")
    read_limit = min(_MAX_SOURCE_FILE_BYTES, aggregate_remaining)
    try:
        with source.open("rb") as handle:
            raw = handle.read(read_limit + 1)
    except OSError as error:
        raise ValueError(f"source bundle file cannot be read: {source.name}") from error
    if len(raw) > _MAX_SOURCE_FILE_BYTES:
        raise ValueError("source bundle file exceeds the per-file byte limit")
    if len(raw) > aggregate_remaining:
        raise ValueError("source bundle exceeds the aggregate byte limit")
    return raw


def build_execution_plan(root: Path, protocol_path: Path) -> dict[str, object]:
    """Return a byte-bindable no-execution plan for the exact local sources."""
    if not isinstance(root, Path) or not isinstance(protocol_path, Path):
        raise TypeError("plan root and protocol path must be Paths")
    protocol = load_composition_protocol(protocol_path)
    native_build_paths = tuple(
        f"src/humanbrain/_native_src/{relative_path}"
        for relative_path, _path in native_runtime._native_build_input_paths()
    )
    missing_native_paths = tuple(
        path for path in native_build_paths if path not in protocol.source_bundle_paths
    )
    if missing_native_paths:
        raise ValueError("source bundle omits a native Cargo build input")
    missing_local_paths = tuple(
        path
        for path in _required_local_source_paths(root)
        if path not in protocol.source_bundle_paths
    )
    if missing_local_paths:
        raise ValueError("source bundle omits a transitive local runtime source")
    bundle: list[dict[str, object]] = []
    aggregate_bytes = 0
    for relative_path in protocol.source_bundle_paths:
        source = _resolved_source(root, relative_path)
        raw = _bounded_source_bytes(source, _MAX_SOURCE_BUNDLE_BYTES - aggregate_bytes)
        aggregate_bytes += len(raw)
        bundle.append(
            {
                "path": relative_path,
                "byte_count": len(raw),
                "sha256": sha256(raw).hexdigest(),
            }
        )
    bundle_sha256 = sha256(_canonical_json(bundle)).hexdigest()
    return {
        "protocol_id": protocol.protocol_id,
        "mode": "plan",
        "material_execution_authorized": False,
        "protocol_path": str(protocol_path.resolve()),
        "protocol_sha256": protocol.raw_sha256,
        "base_commit": protocol.base_commit,
        "source_bundle": bundle,
        "source_bundle_sha256": bundle_sha256,
        "source_bundle_limits": {
            "maximum_file_bytes": _MAX_SOURCE_FILE_BYTES,
            "maximum_aggregate_bytes": _MAX_SOURCE_BUNDLE_BYTES,
            "actual_aggregate_bytes": aggregate_bytes,
        },
        "lanes": [
            {
                "lane": lane.lane,
                "diagnostic_only": lane.diagnostic_only,
                "sole_capability_candidate": lane.eligible,
                "framing_order": lane.framing_order,
                "supervision_scope": lane.supervision_scope,
            }
            for lane in protocol.lanes
        ],
        "accounting": {
            name: getattr(protocol.accounting, name)
            for name in protocol.accounting.__dataclass_fields__
        },
        "future_cohort": {
            "seed_exists": protocol.future_cohort.seed is not None,
            "generated": protocol.future_cohort.generated,
            "task_count": protocol.future_cohort.task_count,
            "factor_counts": protocol.future_cohort.factor_counts,
        },
        "execution_requirements": {
            "protocol_confirmation": "exact raw protocol SHA-256",
            "source_bundle_confirmation": "exact reviewed source-bundle SHA-256",
            "evidence": "one durable exclusive protocol/cohort attempt",
            "resume": False,
            "rerun_same_protocol_cohort": False,
            "preserve_failure": True,
        },
        "run_hold": (
            "No material TRAIN, calibration, or future-cohort execution is "
            "authorized before the protocol/source hashes are reported and the hold "
            "is explicitly released."
        ),
    }


def validate_execution_confirmation(
    protocol: CompositionProtocol,
    confirmation: str | None,
) -> None:
    if confirmation != protocol.raw_sha256:
        raise ProtocolConfirmationError("execution requires the exact raw protocol SHA-256")


def validate_source_bundle_confirmation(
    plan: Mapping[str, object],
    confirmation: str | None,
) -> None:
    expected = plan.get("source_bundle_sha256")
    if not isinstance(expected, str) or confirmation != expected:
        raise ProtocolConfirmationError(
            "execution requires the exact reviewed source-bundle SHA-256"
        )


def _require_cpu_runtime_backend() -> dict[str, object]:
    """Freeze the 128-cell comparison to the same native CPU backend."""
    requested = os.environ.get("HUMANBRAIN_DEVICE", "auto").strip().lower()
    reservoir_cells = typed_program._CIRCUIT_CONFIG.reservoir_cells
    cuda_minimum_cells = cuda_runtime._CUDA_MINIMUM_CELLS
    if requested not in {"auto", "cpu"} or reservoir_cells >= cuda_minimum_cells:
        raise RuntimeError("typed-composition execution requires the frozen native CPU backend")
    return {
        "requested_device": requested,
        "resolved_backend": "native-cpu",
        "reservoir_cells": reservoir_cells,
        "cuda_minimum_cells": cuda_minimum_cells,
    }


def _runtime_provenance() -> dict[str, object]:
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
    }


def _elapsed_ns(start: int, finish: int, label: str) -> int:
    if type(start) is not int or type(finish) is not int or finish < start:
        raise RuntimeError(f"{label} clock did not return monotonic integer nanoseconds")
    return finish - start


def _validate_checkpoint_accounting(
    protocol: CompositionProtocol,
    lane: str,
    checkpoint: object,
) -> dict[str, int]:
    """Bind a grown checkpoint to the exact public input and call accounting."""
    layer_protocol = Layer1Protocol.for_lane(lane)
    digest = sha256()
    input_hashes: list[str] = []
    sequence_counts: list[int] = []
    teacher_counts: list[int] = []
    for task in protocol.train_tasks:
        input_ids = typed_program._model_input(
            task.prompt,
            task.signature,
            framing_order=layer_protocol.framing_order,
        )
        teacher_ids = typed_program._program_token_ids(task.teacher_tokens)
        input_bytes = typed_program._token_stream_bytes(input_ids)
        input_hashes.append(sha256(input_bytes).hexdigest())
        sequence_counts.append(len(input_ids) + len(teacher_ids))
        teacher_counts.append(len(teacher_ids))
        digest.update(input_bytes)
        digest.update(typed_program._token_stream_bytes(teacher_ids))

    expected = (
        lane,
        digest.hexdigest(),
        tuple(input_hashes),
        tuple(sequence_counts),
        tuple(teacher_counts),
        sha256(typed_program._protocol_json(layer_protocol)).hexdigest(),
    )
    actual = (
        getattr(checkpoint, "lane", None),
        getattr(checkpoint, "training_digest", None),
        getattr(checkpoint, "training_input_sha256", None),
        getattr(checkpoint, "training_sequence_token_counts", None),
        getattr(checkpoint, "training_teacher_token_counts", None),
        getattr(checkpoint, "protocol_sha256", None),
    )
    if actual != expected:
        raise ValueError("grown checkpoint accounting differs from the frozen corpus")

    teacher_targets = sum(teacher_counts)
    framing_tokens = sum(
        sequence_count - teacher_count
        for sequence_count, teacher_count in zip(
            sequence_counts,
            teacher_counts,
            strict=True,
        )
    )
    observations = sum(count - 1 for count in sequence_counts)
    passes = layer_protocol.growth_epochs
    strengthen_calls = passes * (
        teacher_targets
        if layer_protocol.supervision_scope == "program-token-targets-only"
        else observations
    )
    accounting = {
        "training_sequences": len(sequence_counts),
        "teacher_targets_per_pass": teacher_targets,
        "framing_tokens_per_pass": framing_tokens,
        "observations_per_pass": observations,
        "corpus_passes": passes,
        "observation_calls": passes * observations,
        "strengthen_calls": strengthen_calls,
    }
    expected_strengthens = (
        protocol.accounting.a2_strengthen_calls
        if lane == "A2"
        else protocol.accounting.a0_a1_strengthen_calls
    )
    if accounting != {
        "training_sequences": protocol.accounting.train_task_count,
        "teacher_targets_per_pass": protocol.accounting.teacher_targets_per_pass,
        "framing_tokens_per_pass": protocol.accounting.framing_tokens_per_pass,
        "observations_per_pass": protocol.accounting.observations_per_pass,
        "corpus_passes": protocol.accounting.corpus_passes,
        "observation_calls": protocol.accounting.observation_calls_per_lane,
        "strengthen_calls": expected_strengthens,
    }:
        raise ValueError("derived checkpoint accounting differs from the protocol")
    return accounting


def create_evidence_run(
    evidence_root: Path,
    run_id: str,
    plan_bytes: bytes,
) -> Path:
    """Exclusively create a never-resumable evidence directory and first record."""
    if not isinstance(evidence_root, Path):
        raise TypeError("evidence root must be a Path")
    if (
        not isinstance(run_id, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", run_id) is None
        or run_id.endswith(".")
    ):
        raise ValueError("run id must be a safe one-to-sixty-four character identifier")
    if type(plan_bytes) is not bytes:
        raise TypeError("plan bytes must be raw bytes")
    evidence_root = _durably_create_missing_directories(evidence_root)
    run_directory = evidence_root / run_id
    try:
        _create_new_directory(run_directory)
    except EvidenceExistsError as error:
        raise EvidenceExistsError(f"evidence run already exists: {run_id}") from error
    except BaseException as error:
        if _is_plain_directory(run_directory):
            _terminal_failure(run_directory, error)
        raise
    plan_path = run_directory / "000-plan.json"
    try:
        _write_new(plan_path, plan_bytes)
    except EvidenceExistsError as error:
        wrapped = EvidenceExistsError("initial evidence record already exists")
        _terminal_failure(run_directory, wrapped)
        raise wrapped from error
    except BaseException as error:
        _terminal_failure(run_directory, error)
        raise
    return run_directory


def _claim_production_attempt(root: Path) -> Path:
    """Durably spend the sole fixed protocol/cohort attempt.

    The direct-create marker has no run-id, protocol-hash, or caller-selected
    evidence-root input. A process interruption at any later point therefore
    still consumes the same one-shot attempt forever.
    """
    run_directory = _production_attempt_directory(root)
    claim_marker = _production_attempt_claim_marker(root)
    staging_directory = _production_attempt_staging_directory(root)
    invalid_marker = _production_attempt_invalid_marker(root)
    complete_marker = _production_attempt_complete_marker(root)
    claim_acquired = False

    def mark_claim_acquired() -> None:
        nonlocal claim_acquired
        claim_acquired = True

    try:
        # The direct-create marker is the first and sole namespace authority.
        # Even a later final/staging collision therefore permanently spends
        # this attempted execution.
        _create_durable_claim_marker(
            claim_marker,
            on_created=mark_claim_acquired,
        )
        if any(
            os.path.lexists(path)
            for path in (
                run_directory,
                staging_directory,
                invalid_marker,
                complete_marker,
            )
        ):
            raise EvidenceExistsError(
                "fixed typed-composition protocol/cohort attempt already exists"
            )
        if os.name == "nt":
            try:
                staging_directory.mkdir(exist_ok=False)
            except FileExistsError as error:
                raise EvidenceExistsError(
                    "fixed typed-composition protocol/cohort attempt already exists"
                ) from error
            # This deterministic directory is itself a permanent spent marker.
            # If interruption occurs before the write-through move, every later
            # claimant observes it and refuses the cohort forever.
            _require_plain_directory(staging_directory)
            _windows_move_new(staging_directory, run_directory)
        else:
            _create_new_directory(run_directory)
        _require_plain_directory(run_directory)
    except EvidenceExistsError as error:
        # Once our permanent marker exists, every visible plain collision is an
        # invalid terminal outcome. Seal both names so neither a stale final
        # envelope nor an indeterminate staging directory can become authority.
        if claim_acquired:
            try:
                _durably_invalidate_production_attempt(root)
            except BaseException as invalidation_error:
                error.add_note(
                    "parent invalid-marker publication also failed: "
                    f"{type(invalidation_error).__name__}: {invalidation_error}"
                )
            for candidate in (staging_directory, run_directory):
                if _is_plain_directory(candidate):
                    _terminal_failure(candidate, error)
        raise EvidenceExistsError(
            "fixed typed-composition protocol/cohort attempt already exists"
        ) from error
    except BaseException as error:
        # MoveFileExW may make the target visible and then report an
        # indeterminate write-through failure. Probe the canonical outcome
        # first, then the deterministic staging outcome.
        if claim_acquired:
            try:
                _durably_invalidate_production_attempt(root)
            except BaseException as invalidation_error:
                error.add_note(
                    "parent invalid-marker publication also failed: "
                    f"{type(invalidation_error).__name__}: {invalidation_error}"
                )
        for candidate in (run_directory, staging_directory):
            if _is_plain_directory(candidate):
                _terminal_failure(candidate, error)
                break
        raise
    return run_directory


def _production_attempt_claim_passes(
    root: Path,
    run_directory: Path,
    claimed_directory_identity: tuple[int, int],
    *,
    allow_completed: bool = False,
) -> bool:
    """Re-derive the fixed claim from its live path and captured identity."""
    if type(allow_completed) is not bool or (
        type(claimed_directory_identity) is not tuple
        or len(claimed_directory_identity) != 2
        or any(type(value) is not int for value in claimed_directory_identity)
    ):
        return False
    try:
        expected = _production_attempt_directory(root)
        marker = _production_attempt_claim_marker(root)
        staging = _production_attempt_staging_directory(root)
        invalid = _production_attempt_invalid_marker(root)
        complete = _production_attempt_complete_marker(root)
        completion_state_passes = (
            _read_bounded_plain_regular_file(
                complete,
                maximum_bytes=len(_PRODUCTION_COMPLETE_MARKER_BYTES),
            )
            == _PRODUCTION_COMPLETE_MARKER_BYTES
            if allow_completed
            else not os.path.lexists(complete)
        )
        return (
            _lexical_absolute(run_directory) == expected
            and _plain_directory_identity(run_directory)
            == claimed_directory_identity
            and not os.path.lexists(staging)
            and not os.path.lexists(invalid)
            and completion_state_passes
            and _read_bounded_plain_regular_file(
                marker,
                maximum_bytes=len(_PRODUCTION_CLAIM_MARKER_BYTES),
            )
            == _PRODUCTION_CLAIM_MARKER_BYTES
            and not os.path.lexists(staging)
            and not os.path.lexists(invalid)
        )
    except (OSError, RuntimeError, ValueError):
        return False


def _race_record(race: ProgramRaceAudit) -> dict[str, object]:
    eligibility = race.eligibility_bytes
    if type(eligibility) is not bytes:
        raise TypeError("race eligibility state must be exact raw bytes")
    return {
        "accepted_token": race.accepted_token,
        "winner_cell_id": race.winner_cell_id,
        "spike_time_ms": race.spike_time_ms,
        "simulated_steps": race.simulated_steps,
        "prefix_sha256": race.prefix_sha256,
        "allowed_cell_ids": list(race.allowed_cell_ids),
        "allowed_output_currents_pa": list(race.allowed_output_currents_pa),
        "excluded_cell_ids": list(race.excluded_cell_ids),
        "output_current_sha256": race.output_current_sha256,
        "eligibility_f64le_base64": base64.b64encode(eligibility).decode("ascii"),
        "eligibility_sha256": sha256(eligibility).hexdigest(),
    }


def _completion_record(completion: Completion) -> dict[str, object]:
    if type(completion) is not Completion:
        raise TypeError("completion boundary must return an exact Completion")
    return {
        "program": {
            "tokens": list(completion.program.tokens),
            "signature": completion.program.signature.to_record(),
            "sha256": completion.program.sha256,
        },
        "races": [_race_record(race) for race in completion.races],
        "prompt_sha256": completion.prompt_sha256,
        "model_input_sha256": completion.model_input_sha256,
        "checkpoint_sha256": completion.checkpoint_sha256,
        "checkpoint_cell_synapse_before": completion.checkpoint_cell_synapse_before,
        "checkpoint_cell_synapse_after": completion.checkpoint_cell_synapse_after,
    }


def seal_all_lane_completions(
    protocol: CompositionProtocol,
    checkpoints: Mapping[str, object],
    *,
    complete_fn: Callable[[object, str, object], Completion] = complete,  # type: ignore[assignment]
) -> tuple[SealedCompletion, ...]:
    """Complete and byte-seal every lane before any scoring function is called."""
    sealed, _measurements = _seal_all_lane_completions(
        protocol,
        checkpoints,
        complete_fn=complete_fn,
        clock=_SYSTEM_CLOCK,
    )
    return sealed


def _seal_all_lane_completions(
    protocol: CompositionProtocol,
    checkpoints: Mapping[str, object],
    *,
    complete_fn: Callable[[object, str, object], Completion],
    clock: _ExecutionClock,
) -> tuple[tuple[SealedCompletion, ...], dict[str, dict[str, int]]]:
    if tuple(checkpoints) != tuple(lane.lane for lane in protocol.lanes):
        raise ValueError("checkpoints must be ordered exactly A0, A1, and A2")
    tasks = (*protocol.train_tasks, *protocol.calibration_tasks)
    requests = tuple(task.completion_request() for task in tasks)
    raw_seals: list[tuple[bytes, str]] = []
    measurements: dict[str, dict[str, int]] = {}
    for lane in protocol.lanes:
        checkpoint = checkpoints[lane.lane]
        if getattr(checkpoint, "lane", None) != lane.lane:
            raise ValueError("checkpoint lane differs from its sealed lane")
        wall_start = clock.wall_ns()
        process_start = clock.process_ns()
        completion_count = 0
        race_count = 0
        for request in requests:
            # This call is the model boundary. Do not add a task, label, case,
            # family, template, ordinal, or evidence identifier to it.
            completion = complete_fn(checkpoint, request.prompt, request.signature)
            record_bytes = _canonical_json(_completion_record(completion))
            raw_seals.append((record_bytes, sha256(record_bytes).hexdigest()))
            completion_count += 1
            race_count += len(completion.races)
        process_finish = clock.process_ns()
        wall_finish = clock.wall_ns()
        measurements[lane.lane] = {
            "completion_sealing_wall_ns": _elapsed_ns(
                wall_start,
                wall_finish,
                "completion wall",
            ),
            "completion_sealing_process_ns": _elapsed_ns(
                process_start,
                process_finish,
                "completion process",
            ),
            "completion_count": completion_count,
            "race_count": race_count,
        }

    # Trusted evidence identities are attached only after every model call has
    # returned and every raw completion record has been byte-sealed.
    sealed: list[SealedCompletion] = []
    sequence_number = 0
    raw_index = 0
    for lane in protocol.lanes:
        for task in tasks:
            record_bytes, record_sha256 = raw_seals[raw_index]
            raw_index += 1
            sequence_number += 1
            sealed.append(
                SealedCompletion(
                    sequence_number=sequence_number,
                    lane=lane.lane,
                    cohort=task.cohort,
                    task_id=task.task_id,
                    record_bytes=record_bytes,
                    sha256=record_sha256,
                )
            )
    return tuple(sealed), measurements


def _a2_capability_gates_pass(lane: str, metrics: A2CapabilityMetrics) -> bool:
    """Report whether lane diagnostics satisfy the frozen capability gates."""
    return (
        lane == "A2"
        and metrics.train_replay_tasks == 42
        and metrics.train_cases == 126
        and metrics.calibration_replay_tasks == 18
        and metrics.calibration_cases == 54
        and metrics.common_integrity_gates_passed
        and metrics.decision_collision_count == 0
        and metrics.competitive_train_margin_count > 0
        and math.isfinite(metrics.minimum_train_target_margin_pa)
        and metrics.minimum_train_target_margin_pa > 0.0
    )


def _terminal_a2_eligibility(
    metrics: A2CapabilityMetrics,
    gates: _ProductionExecutionGates,
) -> bool:
    """Conjoin capability with every authenticated production execution gate."""
    if type(metrics) is not A2CapabilityMetrics:
        raise TypeError("terminal eligibility requires exact A2 capability metrics")
    if type(gates) is not _ProductionExecutionGates:
        raise TypeError("terminal eligibility requires exact production gates")
    return gates.all_passed() and _a2_capability_gates_pass("A2", metrics)


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _aggregate(values: list[float]) -> dict[str, object]:
    return {
        "count": len(values),
        "minimum": min(values) if values else 0.0,
        "p10": _percentile(values, 0.10),
        "median": _percentile(values, 0.50),
        "mean": sum(values) / len(values) if values else 0.0,
    }


def _decode_sealed(item: SealedCompletion) -> dict[str, object]:
    if sha256(item.record_bytes).hexdigest() != item.sha256:
        raise ValueError("sealed completion digest mismatch")
    try:
        value = json.loads(item.record_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("sealed completion is not UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError("sealed completion must be a record")
    return value


def _run_semantic_cases(task: CompositionTask, tokens: tuple[str, ...]) -> int:
    signature = task.signature
    try:
        source = compile(Program(tokens, signature)).source
        namespace: dict[str, object] = {}
        exec(source, {"__builtins__": {}}, namespace)
        function = namespace[signature.entry_point]
    except Exception:
        return 0
    passed = 0
    for case in task.cases:
        try:
            actual = function(*case.arguments)  # type: ignore[operator]
        except Exception:
            continue
        passed += int(actual == case.expected)
    return passed


def _race_integrity(
    races: object,
    tokens: tuple[str, ...],
    signature: TypedSignature,
) -> tuple[bool, list[dict[str, object]]]:
    if not isinstance(races, list) or len(races) != len(tokens):
        return False, []
    if not isinstance(signature, TypedSignature):
        return False, []
    checked: list[dict[str, object]] = []
    state = typed_program._TypedHoleState(signature)
    prefix: list[str] = []
    expected_keys = frozenset(
        {
            "accepted_token",
            "winner_cell_id",
            "spike_time_ms",
            "simulated_steps",
            "prefix_sha256",
            "allowed_cell_ids",
            "allowed_output_currents_pa",
            "excluded_cell_ids",
            "output_current_sha256",
            "eligibility_f64le_base64",
            "eligibility_sha256",
        }
    )
    digest_pattern = re.compile(r"[0-9a-f]{64}")
    for token, value in zip(tokens, races, strict=True):
        if not isinstance(value, dict) or frozenset(value) != expected_keys:
            return False, []
        try:
            allowed_values = value["allowed_cell_ids"]
            current_values = value["allowed_output_currents_pa"]
            excluded_values = value["excluded_cell_ids"]
            if (
                not isinstance(allowed_values, list)
                or not all(type(item) is int for item in allowed_values)
                or not isinstance(current_values, list)
                or not all(type(item) is float for item in current_values)
                or not isinstance(excluded_values, list)
                or not all(type(item) is int for item in excluded_values)
            ):
                return False, []
            allowed = tuple(allowed_values)
            currents = tuple(current_values)
            excluded = tuple(excluded_values)
            winner_value = value["winner_cell_id"]
            spike_time = value["spike_time_ms"]
            simulated_steps = value["simulated_steps"]
            if (
                type(winner_value) is not int
                or type(spike_time) is not float
                or type(simulated_steps) is not int
            ):
                return False, []
            winner = winner_value
            eligibility = base64.b64decode(
                str(value["eligibility_f64le_base64"]).encode("ascii"),
                validate=True,
            )
        except (KeyError, TypeError, ValueError):
            return False, []
        try:
            spec = typed_program._SPEC_BY_TOKEN[token]
        except KeyError:
            return False, []
        expected_allowed = tuple(typed_program._TOKEN_TO_ID[item.token] for item in state.allowed())
        expected_prefix_sha256 = sha256(_canonical_json(prefix)).hexdigest()
        output_current_sha256 = value.get("output_current_sha256")
        prefix_sha256 = value.get("prefix_sha256")
        eligibility_sha256 = value.get("eligibility_sha256")
        if (
            value.get("accepted_token") != token
            or allowed != expected_allowed
            or len(currents) != len(allowed)
            or winner != typed_program._TOKEN_TO_ID[token]
            or len(excluded) != len(set(excluded))
            or any(
                cell_id < 0 or cell_id >= typed_program._TOKEN_CELL_COUNT for cell_id in excluded
            )
            or set(allowed) & set(excluded)
            or not math.isfinite(spike_time)
            or spike_time <= 0.0
            or simulated_steps <= 0
            or prefix_sha256 != expected_prefix_sha256
            or not isinstance(output_current_sha256, str)
            or digest_pattern.fullmatch(output_current_sha256) is None
            or len(eligibility) != 128 * 8
            or eligibility_sha256 != sha256(eligibility).hexdigest()
            or not isinstance(eligibility_sha256, str)
            or digest_pattern.fullmatch(eligibility_sha256) is None
            or not all(math.isfinite(current) for current in currents)
            or not np.all(np.isfinite(np.frombuffer(eligibility, dtype="<f8")))
        ):
            return False, []
        try:
            state.consume(spec)
        except typed_program.ProgramFormatError:
            return False, []
        prefix.append(token)
        checked.append(
            {
                "allowed": allowed,
                "currents": currents,
                "winner": winner,
                "eligibility": eligibility,
            }
        )
    return state.complete, checked


def _state_separation(
    protocol: CompositionProtocol,
    decoded: dict[tuple[str, str], dict[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for lane in ("A0", "A1", "A2"):
        by_type: dict[str, list[tuple[object, np.ndarray]]] = {"Int": [], "List[Int]": []}
        for task in protocol.train_tasks:
            record = decoded[(lane, task.task_id)]
            races = record.get("races")
            if not isinstance(races, list) or not races:
                continue
            first = races[0]
            if not isinstance(first, dict):
                continue
            try:
                raw = base64.b64decode(
                    str(first["eligibility_f64le_base64"]).encode("ascii"),
                    validate=True,
                )
            except (KeyError, ValueError):
                continue
            if len(raw) != 1024:
                continue
            stratum = "List[Int]" if task.family_id.startswith("list_") else "Int"
            by_type[stratum].append((task, np.frombuffer(raw, dtype="<f8")))
        lane_record: dict[str, object] = {}
        for stratum, samples in by_type.items():
            distances: list[float] = []
            for index, (left_task, left) in enumerate(samples):
                for right_task, right in samples[index + 1 :]:
                    if left_task.teacher_tokens[0] == right_task.teacher_tokens[0]:
                        continue
                    left_norm = float(np.linalg.norm(left))
                    right_norm = float(np.linalg.norm(right))
                    if left_norm == 0.0 and right_norm == 0.0:
                        distance = 0.0
                    elif left_norm == 0.0 or right_norm == 0.0:
                        distance = 1.0
                    else:
                        distance = 1.0 - float(np.dot(left, right)) / (left_norm * right_norm)
                    distances.append(distance)
            lane_record[stratum] = _aggregate(distances)
        result[lane] = lane_record
    return result


def _direct_trace_retention(protocol: CompositionProtocol) -> dict[str, object]:
    result: dict[str, object] = {}
    for lane in protocol.lanes:
        strata: dict[str, list[float]] = {"Int": [], "List[Int]": []}
        distances: dict[str, list[int]] = {"Int": [], "List[Int]": []}
        for task in protocol.train_tasks:
            signature_bytes = _canonical_json(task.signature.to_record())
            stratum = "List[Int]" if task.family_id.startswith("list_") else "Int"
            distance = len(signature_bytes) + 2 if lane.lane == "A0" else 1
            distances[stratum].append(distance)
            strata[stratum].append(math.exp(-(distance * 5.0) / 80.0))
        result[lane.lane] = {
            stratum: {
                "token_distances": sorted(set(distances[stratum])),
                "retention": _aggregate(strata[stratum]),
            }
            for stratum in ("Int", "List[Int]")
        }
    return result


@dataclass(frozen=True, slots=True)
class _LaneScore:
    record: dict[str, object]
    integrity_passed: bool
    race_integrity_passed: bool
    checkpoint_cell_synapse_passed: bool
    a2_decisions: tuple[tuple[tuple[int, ...], bytes, int], ...]
    a2_train_margins: tuple[float, ...]
    train_replay_tasks: int
    train_cases: int
    calibration_replay_tasks: int
    calibration_cases: int


def _decode_sealed_records(
    protocol: CompositionProtocol,
    sealed: tuple[SealedCompletion, ...],
) -> tuple[dict[tuple[str, str], dict[str, object]], bool]:
    expected_total = len(protocol.lanes) * (
        len(protocol.train_tasks) + len(protocol.calibration_tasks)
    )
    expected_sequence = tuple(
        (lane.lane, task.cohort, task.task_id)
        for lane in protocol.lanes
        for task in (*protocol.train_tasks, *protocol.calibration_tasks)
    )
    integrity = (
        len(sealed) == expected_total
        and tuple(item.sequence_number for item in sealed)
        == tuple(range(1, expected_total + 1))
        and tuple((item.lane, item.cohort, item.task_id) for item in sealed)
        == expected_sequence
    )
    decoded: dict[tuple[str, str], dict[str, object]] = {}
    for item in sealed:
        key = (item.lane, item.task_id)
        if key in decoded:
            integrity = False
            continue
        try:
            decoded[key] = _decode_sealed(item)
        except ValueError:
            integrity = False
    return decoded, integrity


def _checkpoint_maps_are_valid(
    protocol: CompositionProtocol,
    checkpoint_sha256s: Mapping[str, str],
    checkpoint_cell_synapse_sha256s: Mapping[str, str],
) -> bool:
    expected_lanes = tuple(lane.lane for lane in protocol.lanes)
    digest_pattern = re.compile(r"[0-9a-f]{64}")
    return (
        tuple(checkpoint_sha256s) == expected_lanes
        and tuple(checkpoint_cell_synapse_sha256s) == expected_lanes
        and all(
            isinstance(value, str) and digest_pattern.fullmatch(value) is not None
            for value in (
                *checkpoint_sha256s.values(),
                *checkpoint_cell_synapse_sha256s.values(),
            )
        )
    )


def _score_lane_records(
    protocol: CompositionProtocol,
    lane: LaneCommitment,
    decoded: Mapping[tuple[str, str], dict[str, object]],
    checkpoint_sha256s: Mapping[str, str],
    checkpoint_cell_synapse_sha256s: Mapping[str, str],
) -> _LaneScore:
    train_replay = 0
    train_cases = 0
    calibration_replay = 0
    calibration_cases = 0
    lane_integrity = True
    lane_race_integrity = True
    lane_checkpoint_cell_synapse_integrity = True
    lane_checkpoint_hashes: set[str] = set()
    a2_decisions: list[tuple[tuple[int, ...], bytes, int]] = []
    a2_train_margins: list[float] = []
    digest_pattern = re.compile(r"[0-9a-f]{64}")
    for task in (*protocol.train_tasks, *protocol.calibration_tasks):
        record = decoded.get((lane.lane, task.task_id))
        if record is None or frozenset(record) != frozenset(
            {
                "program",
                "races",
                "prompt_sha256",
                "model_input_sha256",
                "checkpoint_sha256",
                "checkpoint_cell_synapse_before",
                "checkpoint_cell_synapse_after",
            }
        ):
            lane_integrity = False
            lane_race_integrity = False
            lane_checkpoint_cell_synapse_integrity = False
            continue
        program = record.get("program")
        if not isinstance(program, dict) or frozenset(program) != frozenset(
            {"tokens", "signature", "sha256"}
        ):
            lane_integrity = False
            lane_race_integrity = False
            lane_checkpoint_cell_synapse_integrity = False
            continue
        token_values = program.get("tokens")
        if not isinstance(token_values, list) or not all(
            isinstance(token, str) for token in token_values
        ):
            lane_integrity = False
            lane_race_integrity = False
            lane_checkpoint_cell_synapse_integrity = False
            continue
        tokens = tuple(token_values)
        try:
            expected_program_sha256 = Program(tokens, task.signature).sha256
        except (TypeError, ValueError):
            lane_integrity = False
            lane_race_integrity = False
            lane_checkpoint_cell_synapse_integrity = False
            continue
        signature_ok = program.get("signature") == task.signature.to_record()
        program_digest_ok = program.get("sha256") == expected_program_sha256
        prompt_ok = record.get("prompt_sha256") == sha256(
            task.prompt.encode("utf-8")
        ).hexdigest()
        layer_protocol = Layer1Protocol.for_lane(lane.lane)
        expected_model_input_sha256 = sha256(
            typed_program._token_stream_bytes(
                typed_program._model_input(
                    task.prompt,
                    task.signature,
                    framing_order=layer_protocol.framing_order,
                )
            )
        ).hexdigest()
        model_input_ok = record.get("model_input_sha256") == expected_model_input_sha256
        before = record.get("checkpoint_cell_synapse_before")
        after = record.get("checkpoint_cell_synapse_after")
        checkpoint_hash = record.get("checkpoint_sha256")
        if (
            isinstance(checkpoint_hash, str)
            and digest_pattern.fullmatch(checkpoint_hash) is not None
        ):
            lane_checkpoint_hashes.add(checkpoint_hash)
        checkpoint_ok = (
            isinstance(before, str)
            and digest_pattern.fullmatch(before) is not None
            and before == after
            and isinstance(checkpoint_hash, str)
            and digest_pattern.fullmatch(checkpoint_hash) is not None
            and checkpoint_hash == checkpoint_sha256s.get(lane.lane)
            and before == checkpoint_cell_synapse_sha256s.get(lane.lane)
        )
        race_ok, race_values = _race_integrity(
            record.get("races"),
            tokens,
            task.signature,
        )
        lane_race_integrity &= race_ok
        lane_checkpoint_cell_synapse_integrity &= checkpoint_ok
        lane_integrity &= (
            signature_ok
            and program_digest_ok
            and prompt_ok
            and model_input_ok
            and checkpoint_ok
            and race_ok
        )
        replay = tokens == task.teacher_tokens
        semantic_passes = _run_semantic_cases(task, tokens)
        if task.cohort == "TRAIN":
            train_replay += int(replay)
            train_cases += semantic_passes
        else:
            calibration_replay += int(replay)
            calibration_cases += semantic_passes

        if lane.lane == "A2" and replay and race_ok:
            for target_token, race_value in zip(
                task.teacher_tokens,
                race_values,
                strict=True,
            ):
                target = typed_program._TOKEN_TO_ID[target_token]
                allowed = race_value["allowed"]
                currents = race_value["currents"]
                eligibility = race_value["eligibility"]
                assert isinstance(allowed, tuple)
                assert isinstance(currents, tuple)
                assert isinstance(eligibility, bytes)
                a2_decisions.append((allowed, eligibility, target))
                if task.cohort == "TRAIN" and len(allowed) > 1:
                    if target not in allowed:
                        lane_integrity = False
                        continue
                    target_index = allowed.index(target)
                    alternatives = [
                        current
                        for index, current in enumerate(currents)
                        if index != target_index
                    ]
                    margin = float(currents[target_index]) - max(alternatives)
                    if math.isfinite(margin):
                        a2_train_margins.append(margin)
                    else:
                        lane_integrity = False

    lane_integrity &= len(lane_checkpoint_hashes) == 1
    return _LaneScore(
        record={
            "diagnostic_only": lane.diagnostic_only,
            "sole_capability_candidate": lane.eligible,
            "train_replay_tasks": train_replay,
            "train_cases": train_cases,
            "calibration_replay_tasks": calibration_replay,
            "calibration_cases": calibration_cases,
            "integrity_passed": lane_integrity,
        },
        integrity_passed=lane_integrity,
        race_integrity_passed=lane_race_integrity,
        checkpoint_cell_synapse_passed=lane_checkpoint_cell_synapse_integrity,
        a2_decisions=tuple(a2_decisions),
        a2_train_margins=tuple(a2_train_margins),
        train_replay_tasks=train_replay,
        train_cases=train_cases,
        calibration_replay_tasks=calibration_replay,
        calibration_cases=calibration_cases,
    )


def _aggregate_lane_scores(
    protocol: CompositionProtocol,
    sealed_count: int,
    decoded: Mapping[tuple[str, str], dict[str, object]],
    *,
    seal_integrity: bool,
    checkpoint_maps_ok: bool,
    lane_scores: Mapping[str, _LaneScore],
) -> ScoredDiagnostics:
    lane_records = {lane: score.record for lane, score in lane_scores.items()}
    race_integrity = all(
        score.race_integrity_passed for score in lane_scores.values()
    )
    checkpoint_cell_synapse_integrity = all(
        score.checkpoint_cell_synapse_passed for score in lane_scores.values()
    )
    common_integrity = (
        seal_integrity
        and checkpoint_maps_ok
        and all(score.integrity_passed for score in lane_scores.values())
    )
    a2_score = lane_scores["A2"]
    collision_groups: dict[tuple[tuple[int, ...], bytes], set[int]] = {}
    for allowed, eligibility, target in a2_score.a2_decisions:
        collision_groups.setdefault((allowed, eligibility), set()).add(target)
    collision_count = sum(len(targets) > 1 for targets in collision_groups.values())
    margins = a2_score.a2_train_margins
    minimum_margin = min(margins) if margins else 0.0
    metrics = A2CapabilityMetrics(
        train_replay_tasks=a2_score.train_replay_tasks,
        train_cases=a2_score.train_cases,
        calibration_replay_tasks=a2_score.calibration_replay_tasks,
        calibration_cases=a2_score.calibration_cases,
        common_integrity_gates_passed=common_integrity,
        decision_collision_count=collision_count,
        competitive_train_margin_count=len(margins),
        minimum_train_target_margin_pa=minimum_margin,
    )
    record = {
        "protocol_sha256": protocol.raw_sha256,
        "sealed_completion_count": sealed_count,
        "all_completions_sealed_before_scoring": True,
        "lanes": lane_records,
        "a2_capability_diagnostics": {
            "train_replay_tasks": metrics.train_replay_tasks,
            "train_cases": metrics.train_cases,
            "calibration_replay_tasks": metrics.calibration_replay_tasks,
            "calibration_cases": metrics.calibration_cases,
            "common_integrity_gates_passed": metrics.common_integrity_gates_passed,
            "decision_collision_count": metrics.decision_collision_count,
            "competitive_train_margin_count": metrics.competitive_train_margin_count,
            "minimum_train_target_margin_pa": metrics.minimum_train_target_margin_pa,
            "target_margin_distribution": _aggregate(margins),
        },
        "diagnostics": {
            "state_separation": _state_separation(protocol, decoded),
            "direct_prompt_trace_retention": _direct_trace_retention(protocol),
            "a0_a1_hypotheses_are_nonveto": True,
        },
    }
    return ScoredDiagnostics(
        record=record,
        a2_metrics=metrics,
        seal_integrity_passed=seal_integrity,
        race_integrity_passed=race_integrity,
        checkpoint_cell_synapse_passed=checkpoint_cell_synapse_integrity,
        checkpoint_maps_passed=checkpoint_maps_ok,
    )


def score_sealed_completions(
    protocol: CompositionProtocol,
    sealed: tuple[SealedCompletion, ...],
    *,
    checkpoint_sha256s: Mapping[str, str],
    checkpoint_cell_synapse_sha256s: Mapping[str, str],
) -> ScoredDiagnostics:
    """Return diagnostics only after the caller has sealed all lanes.

    This public seam never mints benchmark eligibility. Only the production
    execution boundary may do that after protocol, source, accounting, native,
    and exclusive-attempt checks have passed.
    """
    decoded, seal_integrity = _decode_sealed_records(protocol, sealed)
    checkpoint_maps_ok = _checkpoint_maps_are_valid(
        protocol,
        checkpoint_sha256s,
        checkpoint_cell_synapse_sha256s,
    )
    lane_scores = {
        lane.lane: _score_lane_records(
            protocol,
            lane,
            decoded,
            checkpoint_sha256s,
            checkpoint_cell_synapse_sha256s,
        )
        for lane in protocol.lanes
    }
    return _aggregate_lane_scores(
        protocol,
        len(sealed),
        decoded,
        seal_integrity=seal_integrity,
        checkpoint_maps_ok=checkpoint_maps_ok,
        lane_scores=lane_scores,
    )


def _write_temp_fully(file_descriptor: int, value: bytes) -> None:
    remaining = memoryview(value)
    while remaining:
        written = os.write(file_descriptor, remaining)
        if written <= 0:
            raise OSError("evidence temporary write made no progress")
        remaining = remaining[written:]
    os.fsync(file_descriptor)


def _sync_posix_directory(directory: Path) -> None:
    directory = _require_plain_directory(directory)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        file_descriptor = os.open(directory, flags)
    except OSError as error:
        raise OSError(f"cannot open directory for durable metadata: {directory}") from error
    try:
        os.fsync(file_descriptor)
    finally:
        os.close(file_descriptor)


def _move_file_ex(source: Path, target: Path, flags: int) -> int:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    move_file_ex = kernel32.MoveFileExW
    move_file_ex.argtypes = (ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32)
    move_file_ex.restype = ctypes.c_int
    if move_file_ex(str(source), str(target), flags):
        return 0
    return ctypes.get_last_error()


def _windows_move_new(source: Path, target: Path) -> None:
    # Never follow either parent or the final target.  MoveFileExW with no
    # replace flag remains the authority for the final-name collision.
    source_parent = _require_plain_directory(source.parent)
    target_parent = _require_plain_directory(target.parent)
    if source_parent != target_parent:
        raise ValueError("durable Windows publication requires one physical parent")
    error_code = _move_file_ex(source, target, _MOVEFILE_WRITE_THROUGH)
    if error_code == 0:
        return
    if error_code in _WINDOWS_ALREADY_EXISTS:
        raise EvidenceExistsError(f"evidence target already exists: {target.name}")
    raise ctypes.WinError(error_code)


def _publish_posix_new(temporary_path: Path, target: Path) -> None:
    source_parent = _require_plain_directory(temporary_path.parent)
    target_parent = _require_plain_directory(target.parent)
    if source_parent != target_parent:
        raise ValueError("durable POSIX publication requires one physical parent")
    try:
        os.link(temporary_path, target)
    except FileExistsError as error:
        raise EvidenceExistsError(
            f"evidence target already exists: {target.name}"
        ) from error
    try:
        temporary_path.unlink()
    except OSError as unlink_error:
        try:
            _sync_posix_directory(target.parent)
        except OSError as sync_error:
            unlink_error.add_note(
                f"parent-directory sync also failed: {type(sync_error).__name__}: {sync_error}"
            )
        raise
    _sync_posix_directory(target.parent)


def _publish_new(temporary_path: Path, target: Path) -> None:
    if os.name == "nt":
        _windows_move_new(temporary_path, target)
        return
    _publish_posix_new(temporary_path, target)


def _create_new_directory(path: Path) -> None:
    path = _lexical_absolute(path)
    _require_plain_directory(path.parent)
    if os.name == "nt":
        temporary_path = Path(
            tempfile.mkdtemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        )
        try:
            _windows_move_new(temporary_path, path)
        finally:
            with suppress(OSError):
                temporary_path.rmdir()
        return
    try:
        path.mkdir(exist_ok=False)
    except FileExistsError as error:
        raise EvidenceExistsError(f"evidence directory already exists: {path.name}") from error
    _sync_posix_directory(path.parent)


def _durably_create_missing_directories(path: Path) -> Path:
    resolved = _lexical_absolute(path)
    missing: list[Path] = []
    cursor = resolved
    while not os.path.lexists(cursor):
        missing.append(cursor)
        parent = cursor.parent
        if parent == cursor:
            raise OSError("no existing ancestor for durable evidence directory")
        cursor = parent
    _require_plain_directory(cursor)
    for directory in reversed(missing):
        # A component observed missing must be created by this call.  Accepting
        # a racing winner could return before that winner durably publishes the
        # directory entry, so a collision is an indeterminate failure.
        _create_new_directory(directory)
    return resolved


def _write_new(path: Path, value: bytes) -> None:
    if type(value) is not bytes:
        raise TypeError("evidence value must be exact bytes")
    path = _lexical_absolute(path)
    _require_plain_directory(path.parent)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        try:
            _write_temp_fully(file_descriptor, value)
        except BaseException:
            with suppress(OSError):
                os.close(file_descriptor)
            raise
        else:
            os.close(file_descriptor)
        _publish_new(temporary_path, path)
    finally:
        with suppress(OSError):
            temporary_path.unlink(missing_ok=True)


def _terminal_failure(run_directory: Path, error: BaseException) -> None:
    record = {
        "event": "terminal-failure",
        "error_type": type(error).__name__,
        "message": str(error),
        "partial_evidence_preserved": True,
        "resume_allowed": False,
        "rerun_attempt_allowed": False,
    }
    with suppress(BaseException):
        _write_new(run_directory / "999-terminal-failure.json", _canonical_json(record))


def _record_production_terminal_failure(
    root: Path,
    run_directory: Path,
    error: BaseException,
) -> None:
    """Poison parent authority before best-effort in-directory diagnostics."""
    try:
        _durably_invalidate_production_attempt(root)
    except BaseException as invalidation_error:
        error.add_note(
            "parent invalid-marker publication also failed: "
            f"{type(invalidation_error).__name__}: {invalidation_error}"
        )
    _terminal_failure(run_directory, error)


def _validate_run_id(run_id: str) -> str:
    if (
        not isinstance(run_id, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", run_id) is None
        or run_id.endswith(".")
    ):
        raise ValueError("run id must be a safe one-to-sixty-four character identifier")
    return run_id


def _prepare_execution_inputs(
    root: Path,
    protocol_path: Path,
    *,
    run_id: str,
    confirmation: str | None,
    source_bundle_confirmation: str | None,
    synthetic_test_only: bool,
) -> tuple[
    CompositionProtocol,
    dict[str, object],
    dict[str, object],
    dict[str, object],
    bytes,
]:
    _validate_run_id(run_id)
    if type(synthetic_test_only) is not bool:
        raise TypeError("synthetic-test marker must be a bool")
    protocol = load_composition_protocol(protocol_path)
    validate_execution_confirmation(protocol, confirmation)
    plan = build_execution_plan(root, protocol_path)
    validate_source_bundle_confirmation(plan, source_bundle_confirmation)
    runtime_backend = _require_cpu_runtime_backend()
    runtime_provenance = _runtime_provenance()
    execution_plan = {
        **plan,
        "mode": "synthetic-test" if synthetic_test_only else "execute",
        "material_execution_authorized": not synthetic_test_only,
        "synthetic_test_only": synthetic_test_only,
        "confirmed_protocol_sha256": confirmation,
        "confirmed_source_bundle_sha256": source_bundle_confirmation,
        "runtime_backend": runtime_backend,
        "runtime_provenance": runtime_provenance,
        "run_id": run_id,
    }
    return (
        protocol,
        plan,
        runtime_backend,
        runtime_provenance,
        _canonical_json(execution_plan),
    )


def _verify_production_attempt_namespace(root: Path, run_directory: Path) -> None:
    """Fail closed on a moved or link-redirected fixed attempt namespace."""
    expected = _production_attempt_directory(root)
    if _lexical_absolute(run_directory) != expected:
        raise RuntimeError("production protocol/cohort attempt claim is not canonical")
    _require_plain_directory(run_directory)
    if _production_attempt_directory(root) != expected:
        raise RuntimeError("production protocol/cohort attempt namespace changed")


def _preflight_production_execution(
    root: Path,
    protocol_path: Path,
    *,
    run_id: str,
    confirmation: str | None,
    source_bundle_confirmation: str | None,
) -> _ProductionExecutionRun:
    _require_production_checkout(root, protocol_path)
    (
        protocol,
        plan,
        runtime_backend,
        runtime_provenance,
        execution_plan_bytes,
    ) = _prepare_execution_inputs(
        root,
        protocol_path,
        run_id=run_id,
        confirmation=confirmation,
        source_bundle_confirmation=source_bundle_confirmation,
        synthetic_test_only=False,
    )
    # Everything above is validation or bounded reading.  This fixed directory
    # creation is the first production evidence/native/model side effect.
    run_directory = _claim_production_attempt(root)
    try:
        _verify_production_attempt_namespace(root, run_directory)
        claimed_directory_identity = _plain_directory_identity(run_directory)
        _write_new(run_directory / "000-plan.json", execution_plan_bytes)
        run = _ProductionExecutionRun(
            root=root,
            protocol_path=protocol_path,
            protocol=protocol,
            plan=plan,
            runtime_backend=runtime_backend,
            runtime_provenance=runtime_provenance,
            run_directory=run_directory,
            execution_plan_bytes=execution_plan_bytes,
            confirmed_protocol_sha256=confirmation,
            confirmed_source_bundle_sha256=source_bundle_confirmation,
            claimed_directory_identity=claimed_directory_identity,
        )
    except BaseException as error:
        _record_production_terminal_failure(root, run_directory, error)
        raise
    return run


def _preflight_synthetic_execution(
    root: Path,
    protocol_path: Path,
    *,
    evidence_root: Path,
    run_id: str,
    confirmation: str | None,
    source_bundle_confirmation: str | None,
) -> _SyntheticExecutionRun:
    (
        protocol,
        plan,
        runtime_backend,
        runtime_provenance,
        execution_plan_bytes,
    ) = _prepare_execution_inputs(
        root,
        protocol_path,
        run_id=run_id,
        confirmation=confirmation,
        source_bundle_confirmation=source_bundle_confirmation,
        synthetic_test_only=True,
    )
    candidate = _lexical_absolute(evidence_root / run_id)
    production_attempt = _lexical_absolute(
        _production_repository_root() / _PRODUCTION_ATTEMPT_RELATIVE_PATH
    )
    production_staging = production_attempt.with_name(
        f".{production_attempt.name}.claiming"
    )
    production_claim_marker = production_attempt.with_name(
        f".{production_attempt.name}.claimed"
    )
    production_invalid_marker = production_attempt.with_name(
        f".{production_attempt.name}.invalid"
    )
    production_complete_marker = production_attempt.with_name(
        f".{production_attempt.name}.complete"
    )
    if any(
        _path_overlaps_physical_namespace(candidate, protected)
        for protected in (
            production_attempt,
            production_staging,
            production_claim_marker,
            production_invalid_marker,
            production_complete_marker,
        )
    ):
        raise ValueError(
            "synthetic evidence cannot enter the fixed production attempt namespace"
        )
    run_directory = create_evidence_run(
        evidence_root,
        run_id,
        execution_plan_bytes,
    )
    return _SyntheticExecutionRun(
        root=root,
        protocol_path=protocol_path,
        protocol=protocol,
        plan=plan,
        runtime_backend=runtime_backend,
        runtime_provenance=runtime_provenance,
        run_directory=run_directory,
        execution_plan_bytes=execution_plan_bytes,
        confirmed_protocol_sha256=confirmation,
        confirmed_source_bundle_sha256=source_bundle_confirmation,
    )


def _train_execution_lanes(
    run: _ExecutionRun,
    *,
    grow_fn: Callable[[Layer1Protocol, tuple[object, ...]], object],
    clock: _ExecutionClock,
) -> _TrainedLanes:
    if type(run) not in {_ProductionExecutionRun, _SyntheticExecutionRun}:
        raise TypeError("execution run must have an exact production or synthetic type")
    production = type(run) is _ProductionExecutionRun
    if production and (grow_fn is not grow or clock is not _SYSTEM_CLOCK):
        raise TypeError("production learner is hard-wired and cannot be injected")
    if production:
        _verify_production_attempt_namespace(run.root, run.run_directory)
    checkpoint_bytes: dict[str, bytes] = {}
    checkpoints: dict[str, object] = {}
    checkpoint_hashes: dict[str, str] = {}
    checkpoint_cell_synapse_hashes: dict[str, str] = {}
    checkpoint_accounting: dict[str, dict[str, int]] = {}
    performance_lanes: dict[str, dict[str, int]] = {}
    native_library_identity_before: dict[str, int | str] | None = None
    if production:
        native_library_identity_before = native_runtime.loaded_library_identity().to_record()
        _verify_production_attempt_namespace(run.root, run.run_directory)
        _write_new(
            run.run_directory / "005-native-library-before.json",
            _canonical_json(
                {
                    "event": "native-library-identity-before-model",
                    "identity": native_library_identity_before,
                }
            ),
        )
    examples = tuple(run.protocol.training_examples())
    for lane in run.protocol.lanes:
        grow_wall_start = clock.wall_ns()
        grow_process_start = clock.process_ns()
        checkpoint = grow_fn(Layer1Protocol.for_lane(lane.lane), examples)
        grow_process_finish = clock.process_ns()
        grow_wall_finish = clock.wall_ns()
        if production and type(checkpoint) is not FrozenCheckpoint:
            raise TypeError("production growth must return an exact FrozenCheckpoint")
        if production:
            current_native_identity = native_runtime.loaded_library_identity().to_record()
            if current_native_identity != native_library_identity_before:
                raise RuntimeError("loaded native CPU library changed between lanes")
        if getattr(checkpoint, "lane", None) != lane.lane:
            raise ValueError("grown checkpoint lane differs from requested lane")
        checkpoint_accounting[lane.lane] = _validate_checkpoint_accounting(
            run.protocol,
            lane.lane,
            checkpoint,
        )
        raw = checkpoint.to_bytes()  # type: ignore[attr-defined]
        if type(raw) is not bytes:
            raise TypeError("checkpoint serialization must return exact bytes")
        raw_sha256 = sha256(raw).hexdigest()
        if getattr(checkpoint, "sha256", None) != raw_sha256:
            raise ValueError("checkpoint SHA-256 differs from its raw bytes")
        checkpoint_bytes[lane.lane] = raw
        performance_lanes[lane.lane] = {
            "grow_wall_ns": _elapsed_ns(grow_wall_start, grow_wall_finish, "grow wall"),
            "grow_process_ns": _elapsed_ns(
                grow_process_start,
                grow_process_finish,
                "grow process",
            ),
            "checkpoint_bytes": len(raw),
        }
        checkpoint_hashes[lane.lane] = raw_sha256
        cell_synapse_sha256 = getattr(checkpoint, "cell_synapse_sha256", None)
        if (
            not isinstance(cell_synapse_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", cell_synapse_sha256) is None
        ):
            raise ValueError("checkpoint cell/synapse SHA-256 is invalid")
        checkpoint_cell_synapse_hashes[lane.lane] = cell_synapse_sha256
        checkpoints[lane.lane] = checkpoint
        if production:
            _verify_production_attempt_namespace(run.root, run.run_directory)
        _write_new(run.run_directory / f"010-checkpoint-{lane.lane}.json", raw)
    return _TrainedLanes(
        checkpoint_bytes=checkpoint_bytes,
        checkpoints=checkpoints,
        checkpoint_hashes=checkpoint_hashes,
        checkpoint_cell_synapse_hashes=checkpoint_cell_synapse_hashes,
        checkpoint_accounting=checkpoint_accounting,
        performance_lanes=performance_lanes,
        native_library_identity_before=native_library_identity_before,
    )


def _train_production_execution_lanes(
    run: _ProductionExecutionRun,
) -> _TrainedLanes:
    if type(run) is not _ProductionExecutionRun:
        raise TypeError("training requires an exact production execution run")
    return _train_execution_lanes(
        run,
        grow_fn=grow,  # type: ignore[arg-type]
        clock=_SYSTEM_CLOCK,
    )


def _train_synthetic_execution_lanes(
    run: _SyntheticExecutionRun,
    *,
    grow_fn: Callable[[Layer1Protocol, tuple[object, ...]], object],
    clock: _ExecutionClock,
) -> _TrainedLanes:
    if type(run) is not _SyntheticExecutionRun:
        raise TypeError("training requires an exact synthetic execution run")
    return _train_execution_lanes(run, grow_fn=grow_fn, clock=clock)


def _sealed_completion_record_bytes(
    sealed: tuple[SealedCompletion, ...],
) -> bytes:
    return _canonical_json(
        {
            "event": "all-completions-sealed",
            "scoring_started": False,
            "count": len(sealed),
            "items": [
                {
                    "sequence_number": item.sequence_number,
                    "lane": item.lane,
                    "cohort": item.cohort,
                    "task_id": item.task_id,
                    "record_base64": base64.b64encode(item.record_bytes).decode("ascii"),
                    "sha256": item.sha256,
                }
                for item in sealed
            ],
        }
    )


def _seal_execution_lanes(
    run: _ExecutionRun,
    trained: _TrainedLanes,
    *,
    complete_fn: Callable[[object, str, object], Completion],
    clock: _ExecutionClock,
) -> tuple[SealedCompletion, ...]:
    production = type(run) is _ProductionExecutionRun
    if type(run) not in {_ProductionExecutionRun, _SyntheticExecutionRun}:
        raise TypeError("execution run must have an exact production or synthetic type")
    if production and (complete_fn is not complete or clock is not _SYSTEM_CLOCK):
        raise TypeError("production decoder is hard-wired and cannot be injected")
    if production:
        _verify_production_attempt_namespace(run.root, run.run_directory)
    # No task label or semantic case is inspected until all raw seals exist.
    sealed, completion_measurements = _seal_all_lane_completions(
        run.protocol,
        trained.checkpoints,
        complete_fn=complete_fn,
        clock=clock,
    )
    for lane, lane_measurements in completion_measurements.items():
        trained.performance_lanes[lane].update(lane_measurements)
    _write_new(
        run.run_directory / "020-sealed-completions.json",
        _sealed_completion_record_bytes(sealed),
    )
    for lane, checkpoint in trained.checkpoints.items():
        if checkpoint.to_bytes() != trained.checkpoint_bytes[lane]:  # type: ignore[attr-defined]
            raise RuntimeError("completion changed checkpoint raw bytes")
        if (
            sha256(trained.checkpoint_bytes[lane]).hexdigest()
            != trained.checkpoint_hashes[lane]
        ):
            raise RuntimeError("checkpoint raw-byte digest changed")
    if production:
        _verify_production_attempt_namespace(run.root, run.run_directory)
    return sealed


def _seal_production_execution_lanes(
    run: _ProductionExecutionRun,
    trained: _TrainedLanes,
) -> tuple[SealedCompletion, ...]:
    if type(run) is not _ProductionExecutionRun:
        raise TypeError("sealing requires an exact production execution run")
    return _seal_execution_lanes(
        run,
        trained,
        complete_fn=complete,  # type: ignore[arg-type]
        clock=_SYSTEM_CLOCK,
    )


def _seal_synthetic_execution_lanes(
    run: _SyntheticExecutionRun,
    trained: _TrainedLanes,
    *,
    complete_fn: Callable[[object, str, object], Completion],
    clock: _ExecutionClock,
) -> tuple[SealedCompletion, ...]:
    if type(run) is not _SyntheticExecutionRun:
        raise TypeError("sealing requires an exact synthetic execution run")
    return _seal_execution_lanes(
        run,
        trained,
        complete_fn=complete_fn,
        clock=clock,
    )


def _require_plain_regular_file(path: Path) -> Path:
    absolute = _lexical_absolute(path)
    metadata = os.lstat(absolute)
    attributes = int(getattr(metadata, "st_file_attributes", 0))
    if stat.S_ISLNK(metadata.st_mode) or bool(
        attributes & int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    ):
        raise OSError(
            f"evidence record is a symbolic link or reparse point: {absolute}"
        )
    if not stat.S_ISREG(metadata.st_mode):
        raise OSError(f"evidence record is not a regular file: {absolute}")
    return absolute


def _verify_exact_record(path: Path, expected: bytes) -> None:
    if type(expected) is not bytes:
        raise TypeError("expected evidence record must be exact bytes")
    absolute = _require_plain_regular_file(path)
    if absolute.stat().st_size != len(expected):
        raise RuntimeError(f"evidence record byte count changed: {absolute.name}")
    with absolute.open("rb") as handle:
        actual = handle.read(len(expected) + 1)
    _require_plain_regular_file(absolute)
    if actual != expected:
        raise RuntimeError(f"evidence record bytes changed: {absolute.name}")


def _terminal_failure_absent(run_directory: Path) -> bool:
    """Re-derive the absence of terminal failure evidence at this instant."""
    try:
        directory = _require_plain_directory(run_directory)
    except OSError:
        return False
    return not os.path.lexists(directory / "999-terminal-failure.json")


def _preterminal_evidence_passes(
    run_directory: Path,
    expected_records: Mapping[str, bytes],
) -> bool:
    """Re-derive the exact diagnostics-only evidence envelope."""
    try:
        directory = _require_plain_directory(run_directory)
        if any(
            type(name) is not str
            or Path(name).name != name
            or type(expected) is not bytes
            for name, expected in expected_records.items()
        ):
            return False
        expected_names = frozenset(expected_records)
        actual_names = frozenset(entry.name for entry in directory.iterdir())
        if actual_names != expected_names:
            return False
        for name, expected in expected_records.items():
            _verify_exact_record(directory / name, expected)
        return _terminal_failure_absent(directory)
    except (OSError, RuntimeError, TypeError, ValueError):
        return False


def _verify_preterminal_evidence(
    run: _ProductionExecutionRun,
    expected_records: Mapping[str, bytes],
) -> bool:
    if type(run) is not _ProductionExecutionRun:
        raise TypeError("preterminal evidence verification is production-only")
    if not _production_attempt_claim_passes(
        run.root,
        run.run_directory,
        run.claimed_directory_identity,
    ):
        raise RuntimeError("production fixed-attempt claim changed before terminal seal")
    if not _preterminal_evidence_passes(run.run_directory, expected_records):
        raise RuntimeError("production preterminal evidence file set changed")
    if not _terminal_failure_absent(run.run_directory):
        raise RuntimeError("terminal failure already exists before final publication")
    if not _production_attempt_claim_passes(
        run.root,
        run.run_directory,
        run.claimed_directory_identity,
    ):
        raise RuntimeError("production fixed-attempt claim changed during evidence check")
    return True


def _verify_precommit_terminal_authority(
    run: _ExecutionRun,
    expected_summary: bytes,
    expected_preterminal_records: Mapping[str, bytes],
) -> None:
    production = type(run) is _ProductionExecutionRun
    if type(run) not in {_ProductionExecutionRun, _SyntheticExecutionRun}:
        raise TypeError("terminal verification requires an exact execution run")
    if production and not _production_attempt_claim_passes(
        run.root,
        run.run_directory,
        run.claimed_directory_identity,  # type: ignore[attr-defined]
    ):
        raise RuntimeError("production fixed-attempt claim changed")
    if os.path.lexists(run.run_directory / "999-terminal-failure.json"):
        raise RuntimeError("terminal summary is invalidated by terminal failure evidence")
    expected_names = frozenset(
        (*expected_preterminal_records, "999-terminal-summary.json")
    )
    if frozenset(entry.name for entry in run.run_directory.iterdir()) != expected_names:
        raise RuntimeError("terminal evidence file set changed")
    for name, expected in expected_preterminal_records.items():
        _verify_exact_record(run.run_directory / name, expected)
    _verify_exact_record(
        run.run_directory / "999-terminal-summary.json",
        expected_summary,
    )
    replay_result = _verify_terminal_summary_envelope(
        run.run_directory,
        expected_summary=expected_summary,
        precommit_run=run if production else None,
    )
    if replay_result is not None:
        raise RuntimeError("precommit terminal replay exposed an authority record")
    if production and not _production_attempt_claim_passes(
        run.root,
        run.run_directory,
        run.claimed_directory_identity,  # type: ignore[attr-defined]
    ):
        raise RuntimeError("production fixed-attempt claim changed")


def _hash_plain_regular_file(path: Path, *, maximum_bytes: int) -> str:
    if type(maximum_bytes) is not int or maximum_bytes <= 0:
        raise ValueError("evidence hash limit must be a positive integer")
    absolute = _require_plain_regular_file(path)
    digest = sha256()
    byte_count = 0
    with absolute.open("rb") as handle:
        while True:
            chunk = handle.read(min(1024 * 1024, maximum_bytes - byte_count + 1))
            if not chunk:
                break
            byte_count += len(chunk)
            if byte_count > maximum_bytes:
                raise RuntimeError(f"evidence record exceeds byte limit: {absolute.name}")
            digest.update(chunk)
    _require_plain_regular_file(absolute)
    return digest.hexdigest()


def _read_bounded_plain_regular_file(path: Path, *, maximum_bytes: int) -> bytes:
    if type(maximum_bytes) is not int or maximum_bytes <= 0:
        raise ValueError("evidence read limit must be a positive integer")
    absolute = _require_plain_regular_file(path)
    with absolute.open("rb") as handle:
        raw = handle.read(maximum_bytes + 1)
    if len(raw) > maximum_bytes:
        raise RuntimeError(f"evidence record exceeds byte limit: {absolute.name}")
    _require_plain_regular_file(absolute)
    return raw


def _load_canonical_evidence_object(
    path: Path,
    *,
    maximum_bytes: int,
    label: str,
) -> object:
    raw = _read_bounded_plain_regular_file(path, maximum_bytes=maximum_bytes)
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} is not canonical JSON") from error
    if _canonical_json(value) != raw:
        raise RuntimeError(f"{label} is not canonically encoded")
    return value


def _metrics_from_diagnostics_record(
    diagnostics_record: Mapping[str, object],
) -> A2CapabilityMetrics:
    value = diagnostics_record.get("a2_capability_diagnostics")
    expected_keys = frozenset(
        (*A2CapabilityMetrics.__dataclass_fields__, "target_margin_distribution")
    )
    if not isinstance(value, dict) or frozenset(value) != expected_keys:
        raise RuntimeError("diagnostics evidence has invalid A2 capability metrics")
    try:
        return A2CapabilityMetrics(
            **{
                name: value[name]
                for name in A2CapabilityMetrics.__dataclass_fields__
            }
        )
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            "diagnostics evidence has invalid A2 capability metrics"
        ) from error


def _load_sealed_completion_evidence(
    path: Path,
    protocol: CompositionProtocol,
) -> tuple[SealedCompletion, ...]:
    raw = _read_bounded_plain_regular_file(path, maximum_bytes=64 * 1024 * 1024)
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("sealed-completion evidence is not canonical JSON") from error
    expected_count = len(protocol.lanes) * (
        len(protocol.train_tasks) + len(protocol.calibration_tasks)
    )
    if (
        not isinstance(value, dict)
        or frozenset(value) != frozenset({"event", "scoring_started", "count", "items"})
        or value["event"] != "all-completions-sealed"
        or value["scoring_started"] is not False
        or type(value["count"]) is not int
        or not isinstance(value["items"], list)
        or value["count"] != len(value["items"])
        or value["count"] != expected_count
    ):
        raise RuntimeError("sealed-completion evidence schema is invalid")
    sealed: list[SealedCompletion] = []
    for item in value["items"]:
        if not isinstance(item, dict) or frozenset(item) != frozenset(
            {
                "sequence_number",
                "lane",
                "cohort",
                "task_id",
                "record_base64",
                "sha256",
            }
        ):
            raise RuntimeError("sealed-completion item schema is invalid")
        if (
            type(item["sequence_number"]) is not int
            or not isinstance(item["lane"], str)
            or not isinstance(item["cohort"], str)
            or not isinstance(item["task_id"], str)
            or not isinstance(item["record_base64"], str)
            or not isinstance(item["sha256"], str)
        ):
            raise RuntimeError("sealed-completion item type is invalid")
        try:
            record_bytes = base64.b64decode(item["record_base64"], validate=True)
        except (ValueError, TypeError) as error:
            raise RuntimeError("sealed-completion item base64 is invalid") from error
        if sha256(record_bytes).hexdigest() != item["sha256"]:
            raise RuntimeError("sealed-completion item digest is invalid")
        try:
            decoded_record = json.loads(record_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError(
                "sealed-completion item record is not canonical JSON"
            ) from error
        if _canonical_json(decoded_record) != record_bytes:
            raise RuntimeError(
                "sealed-completion item record is not canonically encoded"
            )
        sealed.append(
            SealedCompletion(
                sequence_number=item["sequence_number"],
                lane=item["lane"],
                cohort=item["cohort"],
                task_id=item["task_id"],
                record_bytes=record_bytes,
                sha256=item["sha256"],
            )
        )
    result = tuple(sealed)
    if _sealed_completion_record_bytes(result) != raw:
        raise RuntimeError("sealed-completion evidence is not canonically encoded")
    return result


def _cpu_backend_record_passes(value: object) -> bool:
    return (
        isinstance(value, dict)
        and frozenset(value)
        == frozenset(
            {
                "requested_device",
                "resolved_backend",
                "reservoir_cells",
                "cuda_minimum_cells",
            }
        )
        and value["requested_device"] in {"auto", "cpu"}
        and value["resolved_backend"] == "native-cpu"
        and type(value["reservoir_cells"]) is int
        and value["reservoir_cells"] == typed_program._CIRCUIT_CONFIG.reservoir_cells
        and type(value["cuda_minimum_cells"]) is int
        and value["cuda_minimum_cells"] == cuda_runtime._CUDA_MINIMUM_CELLS
        and value["reservoir_cells"] < value["cuda_minimum_cells"]
    )


def _runtime_provenance_record_passes(value: object) -> bool:
    if not isinstance(value, dict) or frozenset(value) != frozenset(
        {"python_implementation", "python_version", "numpy_version", "os"}
    ):
        return False
    os_value = value["os"]
    return (
        all(
            isinstance(value[name], str) and bool(value[name])
            for name in ("python_implementation", "python_version", "numpy_version")
        )
        and isinstance(os_value, dict)
        and frozenset(os_value) == frozenset({"system", "release", "machine"})
        and all(isinstance(os_value[name], str) for name in os_value)
    )


def _load_native_identity_evidence(
    path: Path,
    *,
    expected_event: str,
) -> dict[str, int | str]:
    value = _load_canonical_evidence_object(
        path,
        maximum_bytes=1024 * 1024,
        label="native-library identity evidence",
    )
    if (
        not isinstance(value, dict)
        or frozenset(value) != frozenset({"event", "identity"})
        or value["event"] != expected_event
        or not isinstance(value["identity"], dict)
    ):
        raise RuntimeError("native-library identity evidence schema is invalid")
    identity = value["identity"]
    if frozenset(identity) != frozenset(
        {"binary_sha256", "binary_size", "source_digest"}
    ):
        raise RuntimeError("native-library identity evidence schema is invalid")
    if (
        not isinstance(identity["binary_sha256"], str)
        or re.fullmatch(r"[0-9a-f]{64}", identity["binary_sha256"]) is None
        or type(identity["binary_size"]) is not int
        or identity["binary_size"] <= 0
        or not isinstance(identity["source_digest"], str)
        or re.fullmatch(r"[0-9a-f]{64}", identity["source_digest"]) is None
    ):
        raise RuntimeError("native-library identity evidence value is invalid")
    return {
        "binary_sha256": identity["binary_sha256"],
        "binary_size": identity["binary_size"],
        "source_digest": identity["source_digest"],
    }


def _performance_measurements_pass(
    value: object,
    protocol: CompositionProtocol,
    checkpoint_bytes: Mapping[str, bytes],
    sealed: tuple[SealedCompletion, ...],
) -> bool:
    if (
        not isinstance(value, dict)
        or frozenset(value) != frozenset({"capability_gating", "lanes"})
        or value["capability_gating"] is not False
        or not isinstance(value["lanes"], dict)
    ):
        return False
    lanes = value["lanes"]
    expected_lanes = tuple(lane.lane for lane in protocol.lanes)
    if tuple(lanes) != expected_lanes or tuple(checkpoint_bytes) != expected_lanes:
        return False
    race_counts = {lane: 0 for lane in expected_lanes}
    try:
        for item in sealed:
            decoded = _decode_sealed(item)
            races = decoded.get("races")
            if not isinstance(races, list):
                return False
            race_counts[item.lane] += len(races)
    except (KeyError, ValueError):
        return False
    expected_completion_count = len(protocol.train_tasks) + len(
        protocol.calibration_tasks
    )
    integer_names = frozenset(
        {
            "grow_wall_ns",
            "grow_process_ns",
            "checkpoint_bytes",
            "completion_sealing_wall_ns",
            "completion_sealing_process_ns",
            "completion_count",
            "race_count",
        }
    )
    for lane in expected_lanes:
        measurement = lanes.get(lane)
        if (
            not isinstance(measurement, dict)
            or frozenset(measurement) != integer_names
            or any(
                type(measurement[name]) is not int or measurement[name] < 0
                for name in integer_names
            )
            or measurement["checkpoint_bytes"] != len(checkpoint_bytes[lane])
            or measurement["completion_count"] != expected_completion_count
            or measurement["race_count"] != race_counts[lane]
        ):
            return False
    return True


def _loaded_production_attempt_passes(run_directory: Path) -> bool:
    root = _production_repository_root()
    try:
        expected = _production_attempt_directory(root)
        marker = _production_attempt_claim_marker(root)
        staging = _production_attempt_staging_directory(root)
        invalid = _production_attempt_invalid_marker(root)
        complete = _production_attempt_complete_marker(root)
        before = _plain_directory_identity(run_directory)
        return (
            _lexical_absolute(run_directory) == expected
            and os.path.samefile(run_directory, expected)
            and not os.path.lexists(staging)
            and not os.path.lexists(invalid)
            and _read_bounded_plain_regular_file(
                complete,
                maximum_bytes=len(_PRODUCTION_COMPLETE_MARKER_BYTES),
            )
            == _PRODUCTION_COMPLETE_MARKER_BYTES
            and _read_bounded_plain_regular_file(
                marker,
                maximum_bytes=len(_PRODUCTION_CLAIM_MARKER_BYTES),
            )
            == _PRODUCTION_CLAIM_MARKER_BYTES
            and _plain_directory_identity(run_directory) == before
            and not os.path.lexists(staging)
            and not os.path.lexists(invalid)
        )
    except (OSError, RuntimeError, ValueError):
        return False


def _persisted_preterminal_evidence_passes(
    run_directory: Path,
    hashes: Mapping[str, object],
    expected_names: frozenset[str],
) -> bool:
    try:
        if frozenset(hashes) != expected_names:
            return False
        if frozenset(entry.name for entry in run_directory.iterdir()) != frozenset(
            (*expected_names, "999-terminal-summary.json")
        ):
            return False
        for name, expected_digest in hashes.items():
            if (
                not isinstance(expected_digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", expected_digest) is None
                or _hash_plain_regular_file(
                    run_directory / name,
                    maximum_bytes=64 * 1024 * 1024,
                )
                != expected_digest
            ):
                return False
        return _terminal_failure_absent(run_directory)
    except (OSError, RuntimeError, TypeError, ValueError):
        return False


def _production_execution_plan_evidence_passes(
    plan_value: object,
    current_plan: Mapping[str, object],
) -> bool:
    plan_extra_keys = frozenset(
        {
            "mode",
            "material_execution_authorized",
            "synthetic_test_only",
            "confirmed_protocol_sha256",
            "confirmed_source_bundle_sha256",
            "runtime_backend",
            "runtime_provenance",
            "run_id",
        }
    )
    expected_execution_plan = {
        **current_plan,
        "mode": "execute",
        "material_execution_authorized": True,
    }
    return (
        isinstance(plan_value, dict)
        and frozenset(plan_value) == frozenset((*current_plan, *plan_extra_keys))
        and all(
            plan_value[name] == expected
            for name, expected in expected_execution_plan.items()
        )
        and plan_value["synthetic_test_only"] is False
    )


def _rederive_production_terminal_evidence(
    run_directory: Path,
    terminal: Mapping[str, object],
    diagnostics_record: Mapping[str, object],
    hashes: Mapping[str, object],
    expected_names: frozenset[str],
    *,
    precommit_run: _ProductionExecutionRun | None = None,
) -> tuple[A2CapabilityMetrics, _ProductionExecutionGates]:
    """Rebuild every persisted production gate from the evidence records."""
    root = _production_repository_root()
    if precommit_run is not None and (
        type(precommit_run) is not _ProductionExecutionRun
        or _lexical_absolute(precommit_run.root) != root
        or _lexical_absolute(precommit_run.run_directory)
        != _lexical_absolute(run_directory)
    ):
        raise RuntimeError("production precommit authority context is invalid")
    protocol_path = root / _PROTOCOL_RELATIVE_PATH
    protocol = load_composition_protocol(protocol_path)
    current_plan = build_execution_plan(root, protocol_path)
    plan_value = _load_canonical_evidence_object(
        run_directory / "000-plan.json",
        maximum_bytes=8 * 1024 * 1024,
        label="execution plan evidence",
    )
    if not _production_execution_plan_evidence_passes(plan_value, current_plan):
        raise RuntimeError("production execution plan evidence is invalid")
    assert isinstance(plan_value, dict)
    try:
        _validate_run_id(plan_value["run_id"])  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise RuntimeError("production execution metadata run id is invalid") from error
    runtime_backend = plan_value["runtime_backend"]
    runtime_provenance = plan_value["runtime_provenance"]
    native_backend_passed = _cpu_backend_record_passes(runtime_backend)
    if not native_backend_passed or not _runtime_provenance_record_passes(
        runtime_provenance
    ):
        raise RuntimeError("production runtime evidence is invalid")

    native_before = _load_native_identity_evidence(
        run_directory / "005-native-library-before.json",
        expected_event="native-library-identity-before-model",
    )
    native_after = _load_native_identity_evidence(
        run_directory / "025-native-library-after.json",
        expected_event="native-library-identity-before-publish",
    )
    native_library_identity_passed = (
        native_before == native_after
        and terminal["native_library_sha256"] == native_before["binary_sha256"]
    )

    checkpoint_bytes: dict[str, bytes] = {}
    checkpoint_sha256s: dict[str, str] = {}
    checkpoint_cell_synapse_sha256s: dict[str, str] = {}
    checkpoint_accounting: dict[str, dict[str, int]] = {}
    checkpoint_raw_bytes_passed = True
    for lane in protocol.lanes:
        raw = _read_bounded_plain_regular_file(
            run_directory / f"010-checkpoint-{lane.lane}.json",
            maximum_bytes=64 * 1024 * 1024,
        )
        try:
            checkpoint = FrozenCheckpoint.from_bytes(raw)
        except (TypeError, ValueError) as error:
            raise RuntimeError("production checkpoint evidence is invalid") from error
        if checkpoint.lane != lane.lane or checkpoint.to_bytes() != raw:
            checkpoint_raw_bytes_passed = False
        checkpoint_bytes[lane.lane] = raw
        checkpoint_sha256s[lane.lane] = sha256(raw).hexdigest()
        checkpoint_cell_synapse_sha256s[lane.lane] = (
            checkpoint.cell_synapse_sha256
        )
        try:
            checkpoint_accounting[lane.lane] = _validate_checkpoint_accounting(
                protocol,
                lane.lane,
                checkpoint,
            )
        except (TypeError, ValueError) as error:
            raise RuntimeError("production checkpoint accounting is invalid") from error

    sealed = _load_sealed_completion_evidence(
        run_directory / "020-sealed-completions.json",
        protocol,
    )
    diagnostics = score_sealed_completions(
        protocol,
        sealed,
        checkpoint_sha256s=checkpoint_sha256s,
        checkpoint_cell_synapse_sha256s=checkpoint_cell_synapse_sha256s,
    )
    performance_measurements = diagnostics_record.get("performance_measurements")
    completion_attempt_accounting_passed = _performance_measurements_pass(
        performance_measurements,
        protocol,
        checkpoint_bytes,
        sealed,
    )
    observation_strengthen_accounting_passed = all(
        accounting["observation_calls"]
        == protocol.accounting.observation_calls_per_lane
        and accounting["strengthen_calls"]
        == (
            protocol.accounting.a2_strengthen_calls
            if lane == "A2"
            else protocol.accounting.a0_a1_strengthen_calls
        )
        for lane, accounting in checkpoint_accounting.items()
    ) and tuple(checkpoint_accounting) == tuple(
        lane.lane for lane in protocol.lanes
    )
    fixed_attempt_claim_passed = (
        _loaded_production_attempt_passes(run_directory)
        if precommit_run is None
        else _production_attempt_claim_passes(
            root,
            run_directory,
            precommit_run.claimed_directory_identity,
        )
    )
    preterminal_evidence_passed = _persisted_preterminal_evidence_passes(
        run_directory,
        hashes,
        expected_names,
    )
    terminal_failure_absent = _terminal_failure_absent(run_directory)
    protocol_confirmation_passed = (
        plan_value["confirmed_protocol_sha256"] == protocol.raw_sha256
    )
    source_bundle_confirmation_passed = (
        plan_value["confirmed_source_bundle_sha256"]
        == current_plan["source_bundle_sha256"]
    )
    protocol_unchanged = (
        terminal["protocol_sha256"]
        == current_plan["protocol_sha256"]
        == protocol.raw_sha256
    )
    source_bundle_unchanged = (
        terminal["source_bundle_sha256"] == current_plan["source_bundle_sha256"]
    )
    checkpoint_accounting_passed = (
        diagnostics_record.get("checkpoint_accounting") == checkpoint_accounting
    )

    expected_diagnostics = dict(diagnostics.record)
    expected_diagnostics.update(
        {
            "synthetic_test_only": False,
            "source_bundle_sha256": current_plan["source_bundle_sha256"],
            "protocol_unchanged": protocol_unchanged,
            "source_bundle_unchanged": source_bundle_unchanged,
            "runtime_backend": runtime_backend,
            "native_library_identity_before": native_before,
            "native_library_identity_after": native_after,
            "runtime_provenance": runtime_provenance,
            "performance_measurements": performance_measurements,
            "checkpoint_raw_bytes_unchanged": checkpoint_raw_bytes_passed,
            "checkpoint_accounting": checkpoint_accounting,
            "observation_and_strengthen_accounting_passed": (
                observation_strengthen_accounting_passed
            ),
            "exclusive_protocol_cohort_attempt_claimed": (
                fixed_attempt_claim_passed
            ),
        }
    )
    if diagnostics_record != expected_diagnostics:
        raise RuntimeError("production diagnostics differ from rederived evidence")

    gates = _ProductionExecutionGates(
        protocol_confirmation_passed=protocol_confirmation_passed,
        protocol_unchanged=protocol_unchanged,
        source_bundle_confirmation_passed=source_bundle_confirmation_passed,
        source_bundle_unchanged=source_bundle_unchanged,
        fixed_attempt_claim_passed=fixed_attempt_claim_passed,
        native_backend_passed=native_backend_passed,
        native_library_identity_passed=native_library_identity_passed,
        checkpoint_raw_bytes_passed=checkpoint_raw_bytes_passed,
        checkpoint_identity_passed=diagnostics.checkpoint_maps_passed,
        checkpoint_cell_synapse_passed=(
            diagnostics.checkpoint_cell_synapse_passed
        ),
        checkpoint_accounting_passed=checkpoint_accounting_passed,
        observation_strengthen_accounting_passed=(
            observation_strengthen_accounting_passed
        ),
        completion_attempt_accounting_passed=(
            completion_attempt_accounting_passed
        ),
        race_integrity_passed=diagnostics.race_integrity_passed,
        seal_integrity_passed=diagnostics.seal_integrity_passed,
        preterminal_evidence_passed=preterminal_evidence_passed,
        terminal_failure_absent=terminal_failure_absent,
    )
    if not gates.all_passed():
        raise RuntimeError("rederived production execution gate failed")
    return diagnostics.a2_metrics, gates


def _verify_terminal_summary_envelope(
    run_directory: Path,
    *,
    expected_summary: bytes | None = None,
    precommit_run: _ProductionExecutionRun | None = None,
) -> dict[str, object] | None:
    """Replay one envelope; precommit verification never returns its record."""
    if expected_summary is not None and type(expected_summary) is not bytes:
        raise TypeError("expected terminal summary must be exact bytes")
    if precommit_run is not None and expected_summary is None:
        raise TypeError("precommit replay requires expected terminal summary bytes")
    if precommit_run is not None and type(precommit_run) is not _ProductionExecutionRun:
        raise TypeError("precommit replay requires an exact production execution run")
    run_directory = _require_plain_directory(run_directory)
    failure_path = run_directory / "999-terminal-failure.json"
    if os.path.lexists(failure_path):
        raise RuntimeError("terminal summary is invalidated by terminal failure evidence")
    summary_path = run_directory / "999-terminal-summary.json"
    raw = _read_bounded_plain_regular_file(summary_path, maximum_bytes=1024 * 1024)
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("terminal summary is not canonical JSON") from error
    expected_keys = frozenset(
        {
            "event",
            "authority",
            "protocol_sha256",
            "source_bundle_sha256",
            "synthetic_test_only",
            "native_library_sha256",
            "production_execution_gates",
            "all_production_execution_gates_passed",
            "a2_capability_metrics",
            "a2_capability_gates_passed",
            "preterminal_evidence_sha256",
            "a2_eligible",
            "a0_eligible",
            "a1_eligible",
            "future_cohort_generated",
            "resume_allowed",
            "rerun_attempt_allowed",
        }
    )
    if not isinstance(value, dict) or frozenset(value) != expected_keys:
        raise RuntimeError("terminal summary schema is invalid")
    if _canonical_json(value) != raw:
        raise RuntimeError("terminal summary bytes are not canonical")
    if value["event"] != "terminal-summary" or value["authority"] != (
        "sole-final-eligibility-record"
    ):
        raise RuntimeError("terminal summary authority marker is invalid")
    for digest_name in ("protocol_sha256", "source_bundle_sha256"):
        digest_value = value[digest_name]
        if (
            not isinstance(digest_value, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest_value) is None
        ):
            raise RuntimeError(f"terminal summary {digest_name} is invalid")
    metrics_value = value["a2_capability_metrics"]
    if not isinstance(metrics_value, dict) or frozenset(metrics_value) != frozenset(
        A2CapabilityMetrics.__dataclass_fields__
    ):
        raise RuntimeError("terminal A2 capability metrics are invalid")
    try:
        metrics = A2CapabilityMetrics(**metrics_value)
    except (TypeError, ValueError) as error:
        raise RuntimeError("terminal A2 capability metrics are invalid") from error
    synthetic = value["synthetic_test_only"]
    if type(synthetic) is not bool:
        raise RuntimeError("terminal synthetic marker is invalid")
    if precommit_run is not None and synthetic:
        raise RuntimeError("production precommit replay cannot load synthetic evidence")
    gates_value = value["production_execution_gates"]
    gates: _ProductionExecutionGates | None = None
    if synthetic:
        if gates_value is not None or value["native_library_sha256"] is not None:
            raise RuntimeError("synthetic terminal summary asserts production evidence")
        recomputed_eligible = False
        all_gates_passed = False
    else:
        if not isinstance(gates_value, dict) or frozenset(gates_value) != frozenset(
            _ProductionExecutionGates.__dataclass_fields__
        ):
            raise RuntimeError("terminal production execution gates are invalid")
        try:
            gates = _ProductionExecutionGates(**gates_value)
        except TypeError as error:
            raise RuntimeError("terminal production execution gates are invalid") from error
        native_digest = value["native_library_sha256"]
        if (
            not isinstance(native_digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", native_digest) is None
        ):
            raise RuntimeError("terminal native-library SHA-256 is invalid")
        recomputed_eligible = _terminal_a2_eligibility(metrics, gates)
        all_gates_passed = gates.all_passed()
    exact_boolean_fields = (
        "all_production_execution_gates_passed",
        "a2_capability_gates_passed",
        "a2_eligible",
        "a0_eligible",
        "a1_eligible",
        "future_cohort_generated",
        "resume_allowed",
        "rerun_attempt_allowed",
    )
    if any(type(value[name]) is not bool for name in exact_boolean_fields):
        raise RuntimeError("terminal summary boolean field is invalid")
    if value["all_production_execution_gates_passed"] is not all_gates_passed:
        raise RuntimeError("terminal all-gates result is inconsistent")
    if value["a2_capability_gates_passed"] is not _a2_capability_gates_pass(
        "A2", metrics
    ):
        raise RuntimeError("terminal A2 capability result is inconsistent")
    if value["a2_eligible"] is not recomputed_eligible:
        raise RuntimeError("terminal A2 eligibility result is inconsistent")
    if any(
        value[name]
        for name in (
            "a0_eligible",
            "a1_eligible",
            "future_cohort_generated",
            "resume_allowed",
            "rerun_attempt_allowed",
        )
    ):
        raise RuntimeError("terminal fixed-false policy field is invalid")
    hashes = value["preterminal_evidence_sha256"]
    expected_names = frozenset(
        {
            "000-plan.json",
            "010-checkpoint-A0.json",
            "010-checkpoint-A1.json",
            "010-checkpoint-A2.json",
            "020-sealed-completions.json",
            "030-evaluation.json",
            *(
                ()
                if synthetic
                else (
                    "005-native-library-before.json",
                    "025-native-library-after.json",
                )
            ),
        }
    )
    if not isinstance(hashes, dict) or frozenset(hashes) != expected_names:
        raise RuntimeError("terminal preterminal-evidence manifest is invalid")
    if frozenset(entry.name for entry in run_directory.iterdir()) != frozenset(
        (*expected_names, "999-terminal-summary.json")
    ):
        raise RuntimeError("terminal evidence file set is invalid")
    for name, expected_digest in hashes.items():
        if (
            not isinstance(expected_digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected_digest) is None
            or _hash_plain_regular_file(
                run_directory / name,
                maximum_bytes=64 * 1024 * 1024,
            )
            != expected_digest
        ):
            raise RuntimeError(f"terminal evidence digest is invalid: {name}")
    diagnostics_value = _load_canonical_evidence_object(
        run_directory / "030-evaluation.json",
        maximum_bytes=64 * 1024 * 1024,
        label="diagnostics evidence",
    )
    if not isinstance(diagnostics_value, dict):
        raise RuntimeError("diagnostics evidence schema is invalid")
    if _metrics_from_diagnostics_record(diagnostics_value) != metrics:
        raise RuntimeError("terminal metrics differ from diagnostics evidence")
    if not synthetic:
        assert gates is not None
        rederived_metrics, rederived_gates = _rederive_production_terminal_evidence(
            run_directory,
            value,
            diagnostics_value,
            hashes,
            expected_names,
            precommit_run=precommit_run,
        )
        if rederived_metrics != metrics:
            raise RuntimeError("terminal metrics differ from rederived evidence")
        if rederived_gates.to_record() != gates.to_record():
            raise RuntimeError("terminal gates differ from rederived evidence")
        if value["a2_eligible"] is not _terminal_a2_eligibility(
            rederived_metrics,
            rederived_gates,
        ):
            raise RuntimeError("terminal eligibility differs from rederived evidence")
        authority_passed = (
            _loaded_production_attempt_passes(run_directory)
            if precommit_run is None
            else _production_attempt_claim_passes(
                precommit_run.root,
                run_directory,
                precommit_run.claimed_directory_identity,
            )
        )
        if not authority_passed:
            raise RuntimeError("production completion authority changed during load")
    _require_plain_directory(run_directory)
    if expected_summary is not None:
        if _canonical_json(value) != expected_summary:
            raise RuntimeError("terminal replay changed expected summary bytes")
        return None
    return value


def _load_authoritative_terminal_summary(run_directory: Path) -> dict[str, object]:
    """Load only a complete immutable terminal authority; no precommit bypass."""
    value = _verify_terminal_summary_envelope(run_directory)
    if value is None:
        raise RuntimeError("authoritative terminal loader returned no record")
    return value


def load_production_terminal_summary() -> dict[str, object]:
    """Load the sole final production eligibility authority, if it exists."""
    root = _production_repository_root()
    run_directory = _production_attempt_directory(root)
    _verify_production_attempt_namespace(root, run_directory)
    record = _load_authoritative_terminal_summary(run_directory)
    if record["synthetic_test_only"] is not False:
        raise RuntimeError("production terminal summary is marked synthetic")
    protocol_path = root / _PROTOCOL_RELATIVE_PATH
    current_plan = build_execution_plan(root, protocol_path)
    if (
        record["protocol_sha256"] != current_plan["protocol_sha256"]
        or record["source_bundle_sha256"]
        != current_plan["source_bundle_sha256"]
    ):
        raise RuntimeError("terminal summary differs from the current protocol/source")
    _verify_production_attempt_namespace(root, run_directory)
    if not _loaded_production_attempt_passes(run_directory):
        raise RuntimeError("production completion authority changed during final load")
    return record


def _publish_execution_evaluation(
    root: Path,
    protocol_path: Path,
    run: _ExecutionRun,
    trained: _TrainedLanes,
    sealed: tuple[SealedCompletion, ...],
) -> ScoredEvaluation:
    if type(run) not in {_ProductionExecutionRun, _SyntheticExecutionRun}:
        raise TypeError("evaluation publication requires an exact execution run")
    production = type(run) is _ProductionExecutionRun
    diagnostics = score_sealed_completions(
        run.protocol,
        sealed,
        checkpoint_sha256s=trained.checkpoint_hashes,
        checkpoint_cell_synapse_sha256s=trained.checkpoint_cell_synapse_hashes,
    )
    after_plan = build_execution_plan(root, protocol_path)
    protocol_unchanged = after_plan["protocol_sha256"] == run.plan["protocol_sha256"]
    source_bundle_unchanged = (
        after_plan["source_bundle_sha256"] == run.plan["source_bundle_sha256"]
    )
    if not protocol_unchanged or not source_bundle_unchanged:
        raise RuntimeError("protocol or source bundle changed during execution")
    native_library_identity_after: dict[str, int | str] | None = None
    native_before_record: bytes | None = None
    native_after_record: bytes | None = None
    if production:
        assert isinstance(run, _ProductionExecutionRun)
        _verify_production_attempt_namespace(run.root, run.run_directory)
        native_library_identity_after = native_runtime.loaded_library_identity().to_record()
        if native_library_identity_after != trained.native_library_identity_before:
            raise RuntimeError("loaded native CPU library bytes changed during execution")
        native_before_record = _canonical_json(
            {
                "event": "native-library-identity-before-model",
                "identity": trained.native_library_identity_before,
            }
        )
        native_after_record = _canonical_json(
            {
                "event": "native-library-identity-before-publish",
                "identity": native_library_identity_after,
            }
        )
        _write_new(
            run.run_directory / "025-native-library-after.json",
            native_after_record,
        )
    diagnostic_checkpoint_raw_bytes_passed = all(
        checkpoint.to_bytes() == trained.checkpoint_bytes[lane]  # type: ignore[attr-defined]
        and sha256(trained.checkpoint_bytes[lane]).hexdigest()
        == trained.checkpoint_hashes[lane]
        for lane, checkpoint in trained.checkpoints.items()
    ) and frozenset(trained.checkpoint_bytes) == frozenset(
        lane.lane for lane in run.protocol.lanes
    )
    diagnostic_observation_strengthen_passed = all(
        accounting["observation_calls"]
        == run.protocol.accounting.observation_calls_per_lane
        and accounting["strengthen_calls"]
        == (
            run.protocol.accounting.a2_strengthen_calls
            if lane == "A2"
            else run.protocol.accounting.a0_a1_strengthen_calls
        )
        for lane, accounting in trained.checkpoint_accounting.items()
    ) and frozenset(trained.checkpoint_accounting) == frozenset(
        lane.lane for lane in run.protocol.lanes
    )
    diagnostic_attempt_claimed = production and _production_attempt_claim_passes(
        run.root,
        run.run_directory,
        run.claimed_directory_identity,  # type: ignore[attr-defined]
    )
    record = dict(diagnostics.record)
    record["synthetic_test_only"] = not production
    record["source_bundle_sha256"] = run.plan["source_bundle_sha256"]
    record["protocol_unchanged"] = protocol_unchanged
    record["source_bundle_unchanged"] = source_bundle_unchanged
    record["runtime_backend"] = run.runtime_backend
    record["native_library_identity_before"] = (
        trained.native_library_identity_before
    )
    record["native_library_identity_after"] = native_library_identity_after
    record["runtime_provenance"] = run.runtime_provenance
    record["performance_measurements"] = {
        "capability_gating": False,
        "lanes": trained.performance_lanes,
    }
    record["checkpoint_raw_bytes_unchanged"] = (
        diagnostic_checkpoint_raw_bytes_passed
    )
    record["checkpoint_accounting"] = trained.checkpoint_accounting
    record["observation_and_strengthen_accounting_passed"] = (
        diagnostic_observation_strengthen_passed
    )
    record["exclusive_protocol_cohort_attempt_claimed"] = (
        diagnostic_attempt_claimed
    )
    diagnostics_bytes = _canonical_json(record)
    _write_new(
        run.run_directory / "030-evaluation.json",
        diagnostics_bytes,
    )

    capability_metrics_record = {
        name: getattr(diagnostics.a2_metrics, name)
        for name in diagnostics.a2_metrics.__dataclass_fields__
    }
    expected_records = {
        "000-plan.json": run.execution_plan_bytes,
        **{
            f"010-checkpoint-{lane}.json": raw
            for lane, raw in trained.checkpoint_bytes.items()
        },
        "020-sealed-completions.json": _sealed_completion_record_bytes(sealed),
        "030-evaluation.json": diagnostics_bytes,
    }
    gates: _ProductionExecutionGates | None = None
    eligible = False
    if production:
        assert isinstance(run, _ProductionExecutionRun)
        assert native_before_record is not None
        assert native_after_record is not None
        expected_records = {
            **expected_records,
            "005-native-library-before.json": native_before_record,
            "025-native-library-after.json": native_after_record,
        }
        terminal_plan = build_execution_plan(root, protocol_path)
        protocol_unchanged = (
            terminal_plan["protocol_sha256"] == run.plan["protocol_sha256"]
        )
        source_bundle_unchanged = (
            terminal_plan["source_bundle_sha256"]
            == run.plan["source_bundle_sha256"]
        )
        if not protocol_unchanged or not source_bundle_unchanged:
            raise RuntimeError("protocol or source bundle changed before terminal seal")
        terminal_native_identity = native_runtime.loaded_library_identity().to_record()
        if terminal_native_identity != native_library_identity_after:
            raise RuntimeError("loaded native CPU library changed before terminal seal")
        expected_completion_count = len(run.protocol.train_tasks) + len(
            run.protocol.calibration_tasks
        )
        checkpoint_accounting_passed = all(
            _validate_checkpoint_accounting(run.protocol, lane, checkpoint)
            == trained.checkpoint_accounting.get(lane)
            for lane, checkpoint in trained.checkpoints.items()
        ) and frozenset(trained.checkpoint_accounting) == frozenset(
            lane.lane for lane in run.protocol.lanes
        )
        observation_strengthen_accounting_passed = all(
            accounting["observation_calls"]
            == run.protocol.accounting.observation_calls_per_lane
            and accounting["strengthen_calls"]
            == (
                run.protocol.accounting.a2_strengthen_calls
                if lane == "A2"
                else run.protocol.accounting.a0_a1_strengthen_calls
            )
            for lane, accounting in trained.checkpoint_accounting.items()
        )
        checkpoint_raw_bytes_passed = all(
            checkpoint.to_bytes() == trained.checkpoint_bytes[lane]  # type: ignore[attr-defined]
            and sha256(trained.checkpoint_bytes[lane]).hexdigest()
            == trained.checkpoint_hashes[lane]
            for lane, checkpoint in trained.checkpoints.items()
        )
        completion_attempt_accounting_passed = all(
            measurement.get("completion_count") == expected_completion_count
            for measurement in trained.performance_lanes.values()
        ) and frozenset(trained.performance_lanes) == frozenset(
            lane.lane for lane in run.protocol.lanes
        )
        fixed_attempt_claim_passed = _production_attempt_claim_passes(
            run.root,
            run.run_directory,
            run.claimed_directory_identity,
        )
        preterminal_evidence_passed = _verify_preterminal_evidence(
            run,
            expected_records,
        )
        terminal_failure_absent = _terminal_failure_absent(run.run_directory)
        gates = _ProductionExecutionGates(
            protocol_confirmation_passed=(
                run.confirmed_protocol_sha256 == run.protocol.raw_sha256
            ),
            protocol_unchanged=protocol_unchanged,
            source_bundle_confirmation_passed=(
                run.confirmed_source_bundle_sha256
                == run.plan["source_bundle_sha256"]
            ),
            source_bundle_unchanged=source_bundle_unchanged,
            fixed_attempt_claim_passed=fixed_attempt_claim_passed,
            native_backend_passed=(
                run.runtime_backend == _require_cpu_runtime_backend()
            ),
            native_library_identity_passed=(
                trained.native_library_identity_before
                == native_library_identity_after
                == terminal_native_identity
            ),
            checkpoint_raw_bytes_passed=checkpoint_raw_bytes_passed,
            checkpoint_identity_passed=diagnostics.checkpoint_maps_passed,
            checkpoint_cell_synapse_passed=(
                diagnostics.checkpoint_cell_synapse_passed
            ),
            checkpoint_accounting_passed=checkpoint_accounting_passed,
            observation_strengthen_accounting_passed=(
                observation_strengthen_accounting_passed
            ),
            completion_attempt_accounting_passed=(
                completion_attempt_accounting_passed
            ),
            race_integrity_passed=diagnostics.race_integrity_passed,
            seal_integrity_passed=diagnostics.seal_integrity_passed,
            preterminal_evidence_passed=preterminal_evidence_passed,
            terminal_failure_absent=terminal_failure_absent,
        )
        if not gates.all_passed():
            raise RuntimeError("one or more terminal production gates failed")
        eligible = _terminal_a2_eligibility(diagnostics.a2_metrics, gates)

    summary = {
        "event": "terminal-summary",
        "authority": "sole-final-eligibility-record",
        "protocol_sha256": run.protocol.raw_sha256,
        "source_bundle_sha256": run.plan["source_bundle_sha256"],
        "synthetic_test_only": not production,
        "native_library_sha256": (
            None
            if native_library_identity_after is None
            else native_library_identity_after["binary_sha256"]
        ),
        "production_execution_gates": None if gates is None else gates.to_record(),
        "all_production_execution_gates_passed": (
            False if gates is None else gates.all_passed()
        ),
        "a2_capability_metrics": capability_metrics_record,
        "a2_capability_gates_passed": _a2_capability_gates_pass(
            "A2", diagnostics.a2_metrics
        ),
        "preterminal_evidence_sha256": {
            name: sha256(value).hexdigest()
            for name, value in sorted(expected_records.items())
        },
        "a2_eligible": eligible,
        "a0_eligible": False,
        "a1_eligible": False,
        "future_cohort_generated": False,
        "resume_allowed": False,
        "rerun_attempt_allowed": False,
    }
    summary_bytes = _canonical_json(summary)
    result = ScoredEvaluation(
        record=record,
        terminal_summary=summary,
        a2_metrics=diagnostics.a2_metrics,
        a2_eligible=eligible,
    )
    _write_new(
        run.run_directory / "999-terminal-summary.json",
        summary_bytes,
    )
    _verify_precommit_terminal_authority(run, summary_bytes, expected_records)
    if production:
        # This is the final commit action. Public readers require this marker,
        # so no production eligibility authority exists during the fallible
        # terminal replay above. Return the preconstructed result immediately.
        _durably_complete_production_attempt(run.root)
    return result


def _publish_production_execution_evaluation(
    run: _ProductionExecutionRun,
    trained: _TrainedLanes,
    sealed: tuple[SealedCompletion, ...],
) -> ScoredEvaluation:
    if type(run) is not _ProductionExecutionRun:
        raise TypeError("publication requires an exact production execution run")
    return _publish_execution_evaluation(
        run.root,
        run.protocol_path,
        run,
        trained,
        sealed,
    )


def _publish_synthetic_execution_evaluation(
    run: _SyntheticExecutionRun,
    trained: _TrainedLanes,
    sealed: tuple[SealedCompletion, ...],
) -> ScoredEvaluation:
    if type(run) is not _SyntheticExecutionRun:
        raise TypeError("publication requires an exact synthetic execution run")
    result = _publish_execution_evaluation(
        run.root,
        run.protocol_path,
        run,
        trained,
        sealed,
    )
    if result.a2_eligible or result.terminal_summary["a2_eligible"]:
        raise RuntimeError("synthetic publication attempted to mint eligibility")
    return result


def _execute_production_claimed_run(
    run: _ProductionExecutionRun,
) -> ScoredEvaluation:
    """Execute a claimed production run through only the frozen public path."""
    if type(run) is not _ProductionExecutionRun:
        raise TypeError("production executor requires an exact production execution run")
    try:
        trained = _train_production_execution_lanes(run)
        sealed = _seal_production_execution_lanes(run, trained)
        return _publish_production_execution_evaluation(
            run,
            trained,
            sealed,
        )
    except BaseException as error:
        _record_production_terminal_failure(run.root, run.run_directory, error)
        raise


def _execute_synthetic_claimed_run(
    run: _SyntheticExecutionRun,
    *,
    grow_fn: Callable[[Layer1Protocol, tuple[object, ...]], object],
    complete_fn: Callable[[object, str, object], Completion],
    clock: _ExecutionClock,
) -> ScoredEvaluation:
    """Execute an exact synthetic run through injectable test-only seams."""
    if type(run) is not _SyntheticExecutionRun:
        raise TypeError("synthetic executor requires an exact synthetic execution run")
    try:
        trained = _train_synthetic_execution_lanes(
            run,
            grow_fn=grow_fn,
            clock=clock,
        )
        sealed = _seal_synthetic_execution_lanes(
            run,
            trained,
            complete_fn=complete_fn,
            clock=clock,
        )
        return _publish_synthetic_execution_evaluation(
            run,
            trained,
            sealed,
        )
    except BaseException as error:
        _terminal_failure(run.run_directory, error)
        raise


def _execute_protocol(
    root: Path,
    protocol_path: Path,
    *,
    run_id: str,
    confirmation: str | None,
    source_bundle_confirmation: str | None,
) -> ScoredEvaluation:
    """Execute the fixed production path with no injectable synthetic seam."""
    run = _preflight_production_execution(
        root,
        protocol_path,
        run_id=run_id,
        confirmation=confirmation,
        source_bundle_confirmation=source_bundle_confirmation,
    )
    return _execute_production_claimed_run(run)


def execute_protocol(
    protocol_path: Path,
    *,
    run_id: str,
    confirmation: str | None,
    source_bundle_confirmation: str | None,
) -> ScoredEvaluation:
    """Execute the exact public learner and decoder once into exclusive evidence."""
    root = _production_repository_root()
    _require_production_checkout(root, protocol_path)
    return _execute_protocol(
        root,
        protocol_path,
        run_id=run_id,
        confirmation=confirmation,
        source_bundle_confirmation=source_bundle_confirmation,
    )


def _execute_protocol_synthetic(
    root: Path,
    protocol_path: Path,
    *,
    evidence_root: Path,
    run_id: str,
    confirmation: str | None,
    source_bundle_confirmation: str | None,
    grow_fn: Callable[[Layer1Protocol, tuple[object, ...]], object],
    complete_fn: Callable[[object, str, object], Completion] = complete,  # type: ignore[assignment]
    clock: _ExecutionClock = _SYSTEM_CLOCK,
) -> ScoredEvaluation:
    """Exercise failure/evidence plumbing; results are forcibly ineligible."""
    run = _preflight_synthetic_execution(
        root,
        protocol_path,
        evidence_root=evidence_root,
        run_id=run_id,
        confirmation=confirmation,
        source_bundle_confirmation=source_bundle_confirmation,
    )
    return _execute_synthetic_claimed_run(
        run,
        grow_fn=grow_fn,
        complete_fn=complete_fn,
        clock=clock,
    )
