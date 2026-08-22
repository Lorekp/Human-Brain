"""Plan, seal, score, and evidence the frozen typed-composition protocol."""

from __future__ import annotations

import base64
import ctypes
import json
import math
import os
import platform
import re
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
_MAX_SOURCE_BUNDLE_BYTES = 8 * 1024 * 1024
_MOVEFILE_WRITE_THROUGH = 0x00000008
_WINDOWS_ALREADY_EXISTS = frozenset({80, 183})
_PROTOCOL_RELATIVE_PATH = Path("benchmarks/typed_composition_protocol_v1.json")
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
class A2EligibilityMetrics:
    train_replay_tasks: int
    train_cases: int
    calibration_replay_tasks: int
    calibration_cases: int
    common_integrity_gates_passed: bool
    decision_collision_count: int
    competitive_train_margin_count: int
    minimum_train_target_margin_pa: float


@dataclass(frozen=True, slots=True)
class ScoredEvaluation:
    record: dict[str, object]
    a2_metrics: A2EligibilityMetrics
    a2_eligible: bool


@dataclass(frozen=True, slots=True)
class _ExecutionClock:
    wall_ns: Callable[[], int]
    process_ns: Callable[[], int]


@dataclass(frozen=True, slots=True)
class _ExecutionRun:
    protocol: CompositionProtocol
    plan: dict[str, object]
    runtime_backend: dict[str, object]
    runtime_provenance: dict[str, object]
    run_directory: Path


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
                "eligible": lane.eligible,
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
            "evidence": "exclusive nonoverwrite creation",
            "resume": False,
            "rerun_same_run_id": False,
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
        if run_directory.is_dir():
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


def a2_is_eligible(lane: str, metrics: A2EligibilityMetrics) -> bool:
    """Apply the frozen capability gate; A0/A1 can never become eligible."""
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
            continue
        program = record.get("program")
        if not isinstance(program, dict) or frozenset(program) != frozenset(
            {"tokens", "signature", "sha256"}
        ):
            lane_integrity = False
            continue
        token_values = program.get("tokens")
        if not isinstance(token_values, list) or not all(
            isinstance(token, str) for token in token_values
        ):
            lane_integrity = False
            continue
        tokens = tuple(token_values)
        try:
            expected_program_sha256 = Program(tokens, task.signature).sha256
        except (TypeError, ValueError):
            lane_integrity = False
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
            "eligible_lane": lane.eligible,
            "train_replay_tasks": train_replay,
            "train_cases": train_cases,
            "calibration_replay_tasks": calibration_replay,
            "calibration_cases": calibration_cases,
            "integrity_passed": lane_integrity,
        },
        integrity_passed=lane_integrity,
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
) -> ScoredEvaluation:
    lane_records = {lane: score.record for lane, score in lane_scores.items()}
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
    metrics = A2EligibilityMetrics(
        train_replay_tasks=a2_score.train_replay_tasks,
        train_cases=a2_score.train_cases,
        calibration_replay_tasks=a2_score.calibration_replay_tasks,
        calibration_cases=a2_score.calibration_cases,
        common_integrity_gates_passed=common_integrity,
        decision_collision_count=collision_count,
        competitive_train_margin_count=len(margins),
        minimum_train_target_margin_pa=minimum_margin,
    )
    eligible = a2_is_eligible("A2", metrics)
    record = {
        "protocol_sha256": protocol.raw_sha256,
        "sealed_completion_count": sealed_count,
        "all_completions_sealed_before_scoring": True,
        "lanes": lane_records,
        "a2_eligibility": {
            "eligible": eligible,
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
    return ScoredEvaluation(record=record, a2_metrics=metrics, a2_eligible=eligible)


def score_sealed_completions(
    protocol: CompositionProtocol,
    sealed: tuple[SealedCompletion, ...],
    *,
    checkpoint_sha256s: Mapping[str, str],
    checkpoint_cell_synapse_sha256s: Mapping[str, str],
) -> ScoredEvaluation:
    """Score labels and semantics only after the caller has sealed all lanes."""
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
    # Resolve only the parents.  Following the final target component would let
    # an existing reparse point redirect this preflight away from MoveFileExW,
    # whose no-replace result is the authority for exclusive publication.
    if source.parent.resolve() != target.parent.resolve():
        raise ValueError("durable Windows publication requires one resolved parent")
    error_code = _move_file_ex(source, target, _MOVEFILE_WRITE_THROUGH)
    if error_code == 0:
        return
    if error_code in _WINDOWS_ALREADY_EXISTS:
        raise EvidenceExistsError(f"evidence target already exists: {target.name}")
    raise ctypes.WinError(error_code)


def _publish_posix_new(temporary_path: Path, target: Path) -> None:
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
    if not path.parent.is_dir():
        raise OSError(f"durable directory parent does not exist: {path.parent}")
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
    resolved = path.resolve()
    missing: list[Path] = []
    cursor = resolved
    while not cursor.exists():
        missing.append(cursor)
        parent = cursor.parent
        if parent == cursor:
            raise OSError("no existing ancestor for durable evidence directory")
        cursor = parent
    if not cursor.is_dir():
        raise NotADirectoryError(f"evidence ancestor is not a directory: {cursor}")
    for directory in reversed(missing):
        # A component observed missing must be created by this call.  Accepting
        # a racing winner could return before that winner durably publishes the
        # directory entry, so a collision is an indeterminate failure.
        _create_new_directory(directory)
    return resolved


def _write_new(path: Path, value: bytes) -> None:
    if type(value) is not bytes:
        raise TypeError("evidence value must be exact bytes")
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
        "rerun_same_run_id_allowed": False,
    }
    with suppress(BaseException):
        _write_new(run_directory / "999-terminal-failure.json", _canonical_json(record))


def _preflight_execution(
    root: Path,
    protocol_path: Path,
    *,
    evidence_root: Path,
    run_id: str,
    confirmation: str | None,
    source_bundle_confirmation: str | None,
    grow_fn: Callable[[Layer1Protocol, tuple[object, ...]], object],
    complete_fn: Callable[[object, str, object], Completion],
    synthetic_test_only: bool,
    clock: _ExecutionClock,
) -> _ExecutionRun:
    if not synthetic_test_only and (grow_fn is not grow or complete_fn is not complete):
        raise TypeError("production execution requires fixed public grow and complete")
    if not synthetic_test_only and clock is not _SYSTEM_CLOCK:
        raise TypeError("production execution requires the system execution clock")
    if not synthetic_test_only:
        _require_production_checkout(root, protocol_path)
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
    run_directory = create_evidence_run(
        evidence_root,
        run_id,
        _canonical_json(execution_plan),
    )
    return _ExecutionRun(
        protocol=protocol,
        plan=plan,
        runtime_backend=runtime_backend,
        runtime_provenance=runtime_provenance,
        run_directory=run_directory,
    )


def _train_execution_lanes(
    run: _ExecutionRun,
    *,
    grow_fn: Callable[[Layer1Protocol, tuple[object, ...]], object],
    synthetic_test_only: bool,
    clock: _ExecutionClock,
) -> _TrainedLanes:
    checkpoint_bytes: dict[str, bytes] = {}
    checkpoints: dict[str, object] = {}
    checkpoint_hashes: dict[str, str] = {}
    checkpoint_cell_synapse_hashes: dict[str, str] = {}
    checkpoint_accounting: dict[str, dict[str, int]] = {}
    performance_lanes: dict[str, dict[str, int]] = {}
    native_library_identity_before: dict[str, int | str] | None = None
    if not synthetic_test_only:
        native_library_identity_before = native_runtime.loaded_library_identity().to_record()
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
        if not synthetic_test_only and type(checkpoint) is not FrozenCheckpoint:
            raise TypeError("production growth must return an exact FrozenCheckpoint")
        if not synthetic_test_only:
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


def _seal_execution_lanes(
    run: _ExecutionRun,
    trained: _TrainedLanes,
    *,
    complete_fn: Callable[[object, str, object], Completion],
    clock: _ExecutionClock,
) -> tuple[SealedCompletion, ...]:
    # No task label or semantic case is inspected until all raw seals exist.
    sealed, completion_measurements = _seal_all_lane_completions(
        run.protocol,
        trained.checkpoints,
        complete_fn=complete_fn,
        clock=clock,
    )
    for lane, lane_measurements in completion_measurements.items():
        trained.performance_lanes[lane].update(lane_measurements)
    sealed_record = {
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
    _write_new(
        run.run_directory / "020-sealed-completions.json",
        _canonical_json(sealed_record),
    )
    for lane, checkpoint in trained.checkpoints.items():
        if checkpoint.to_bytes() != trained.checkpoint_bytes[lane]:  # type: ignore[attr-defined]
            raise RuntimeError("completion changed checkpoint raw bytes")
        if (
            sha256(trained.checkpoint_bytes[lane]).hexdigest()
            != trained.checkpoint_hashes[lane]
        ):
            raise RuntimeError("checkpoint raw-byte digest changed")
    return sealed


def _publish_execution_evaluation(
    root: Path,
    protocol_path: Path,
    run: _ExecutionRun,
    trained: _TrainedLanes,
    sealed: tuple[SealedCompletion, ...],
    *,
    synthetic_test_only: bool,
) -> ScoredEvaluation:
    evaluation = score_sealed_completions(
        run.protocol,
        sealed,
        checkpoint_sha256s=trained.checkpoint_hashes,
        checkpoint_cell_synapse_sha256s=trained.checkpoint_cell_synapse_hashes,
    )
    if synthetic_test_only:
        evaluation.record["synthetic_test_only"] = True
        evaluation.record["a2_eligibility"]["eligible"] = False  # type: ignore[index]
        evaluation = ScoredEvaluation(
            record=evaluation.record,
            a2_metrics=evaluation.a2_metrics,
            a2_eligible=False,
        )
    after_plan = build_execution_plan(root, protocol_path)
    if (
        after_plan["protocol_sha256"] != run.plan["protocol_sha256"]
        or after_plan["source_bundle_sha256"] != run.plan["source_bundle_sha256"]
    ):
        raise RuntimeError("protocol or source bundle changed during execution")
    native_library_identity_after: dict[str, int | str] | None = None
    if not synthetic_test_only:
        native_library_identity_after = native_runtime.loaded_library_identity().to_record()
        if native_library_identity_after != trained.native_library_identity_before:
            raise RuntimeError("loaded native CPU library bytes changed during execution")
        _write_new(
            run.run_directory / "025-native-library-after.json",
            _canonical_json(
                {
                    "event": "native-library-identity-before-publish",
                    "identity": native_library_identity_after,
                }
            ),
        )
    evaluation.record["source_bundle_sha256"] = run.plan["source_bundle_sha256"]
    evaluation.record["source_bundle_unchanged"] = True
    evaluation.record["runtime_backend"] = run.runtime_backend
    evaluation.record["native_library_identity_before"] = (
        trained.native_library_identity_before
    )
    evaluation.record["native_library_identity_after"] = native_library_identity_after
    evaluation.record["runtime_provenance"] = run.runtime_provenance
    evaluation.record["performance_measurements"] = {
        "capability_gating": False,
        "lanes": trained.performance_lanes,
    }
    evaluation.record["checkpoint_raw_bytes_unchanged"] = True
    evaluation.record["checkpoint_accounting"] = trained.checkpoint_accounting
    evaluation.record["observation_and_strengthen_accounting_passed"] = True
    evaluation.record["exclusive_evidence_creation"] = True
    _write_new(
        run.run_directory / "030-evaluation.json",
        _canonical_json(evaluation.record),
    )
    summary = {
        "event": "terminal-summary",
        "protocol_sha256": run.protocol.raw_sha256,
        "source_bundle_sha256": run.plan["source_bundle_sha256"],
        "synthetic_test_only": synthetic_test_only,
        "native_library_sha256": (
            None
            if native_library_identity_after is None
            else native_library_identity_after["binary_sha256"]
        ),
        "a2_eligible": evaluation.a2_eligible,
        "a0_eligible": False,
        "a1_eligible": False,
        "future_cohort_generated": False,
        "resume_allowed": False,
        "rerun_same_run_id_allowed": False,
    }
    _write_new(
        run.run_directory / "999-terminal-summary.json",
        _canonical_json(summary),
    )
    return evaluation


def _execute_protocol(
    root: Path,
    protocol_path: Path,
    *,
    evidence_root: Path,
    run_id: str,
    confirmation: str | None,
    source_bundle_confirmation: str | None,
    grow_fn: Callable[[Layer1Protocol, tuple[object, ...]], object],
    complete_fn: Callable[[object, str, object], Completion],
    synthetic_test_only: bool,
    clock: _ExecutionClock = _SYSTEM_CLOCK,
) -> ScoredEvaluation:
    """Run the explicit preflight, train, seal, and publish phases once."""
    run = _preflight_execution(
        root,
        protocol_path,
        evidence_root=evidence_root,
        run_id=run_id,
        confirmation=confirmation,
        source_bundle_confirmation=source_bundle_confirmation,
        grow_fn=grow_fn,
        complete_fn=complete_fn,
        synthetic_test_only=synthetic_test_only,
        clock=clock,
    )
    try:
        trained = _train_execution_lanes(
            run,
            grow_fn=grow_fn,
            synthetic_test_only=synthetic_test_only,
            clock=clock,
        )
        sealed = _seal_execution_lanes(
            run,
            trained,
            complete_fn=complete_fn,
            clock=clock,
        )
        return _publish_execution_evaluation(
            root,
            protocol_path,
            run,
            trained,
            sealed,
            synthetic_test_only=synthetic_test_only,
        )
    except BaseException as error:
        _terminal_failure(run.run_directory, error)
        raise

def execute_protocol(
    protocol_path: Path,
    *,
    evidence_root: Path,
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
        evidence_root=evidence_root,
        run_id=run_id,
        confirmation=confirmation,
        source_bundle_confirmation=source_bundle_confirmation,
        grow_fn=grow,  # type: ignore[arg-type]
        complete_fn=complete,  # type: ignore[arg-type]
        synthetic_test_only=False,
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
    return _execute_protocol(
        root,
        protocol_path,
        evidence_root=evidence_root,
        run_id=run_id,
        confirmation=confirmation,
        source_bundle_confirmation=source_bundle_confirmation,
        grow_fn=grow_fn,
        complete_fn=complete_fn,
        synthetic_test_only=True,
        clock=clock,
    )
