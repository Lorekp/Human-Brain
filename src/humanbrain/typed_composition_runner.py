"""Plan, seal, score, and evidence the frozen typed-composition protocol."""

from __future__ import annotations

import base64
import json
import math
import os
import re
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np

from humanbrain import cuda as cuda_runtime
from humanbrain import native as native_runtime
from humanbrain import typed_program
from humanbrain.typed_composition import (
    CompositionProtocol,
    CompositionTask,
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


def build_execution_plan(root: Path, protocol_path: Path) -> dict[str, object]:
    """Return a byte-bindable no-execution plan for the exact local sources."""
    if not isinstance(root, Path) or not isinstance(protocol_path, Path):
        raise TypeError("plan root and protocol path must be Paths")
    protocol = load_composition_protocol(protocol_path)
    bundle: list[dict[str, object]] = []
    for relative_path in protocol.source_bundle_paths:
        source = _resolved_source(root, relative_path)
        raw = source.read_bytes()
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
    evidence_root.mkdir(parents=True, exist_ok=True)
    run_directory = evidence_root / run_id
    try:
        run_directory.mkdir(exist_ok=False)
    except FileExistsError as error:
        raise EvidenceExistsError(f"evidence run already exists: {run_id}") from error
    plan_path = run_directory / "000-plan.json"
    try:
        with plan_path.open("xb") as handle:
            handle.write(plan_bytes)
    except FileExistsError as error:
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
    if tuple(checkpoints) != tuple(lane.lane for lane in protocol.lanes):
        raise ValueError("checkpoints must be ordered exactly A0, A1, and A2")
    tasks = (*protocol.train_tasks, *protocol.calibration_tasks)
    requests = tuple(task.completion_request() for task in tasks)
    raw_seals: list[tuple[bytes, str]] = []
    for lane in protocol.lanes:
        checkpoint = checkpoints[lane.lane]
        if getattr(checkpoint, "lane", None) != lane.lane:
            raise ValueError("checkpoint lane differs from its sealed lane")
        for request in requests:
            # This call is the model boundary. Do not add a task, label, case,
            # family, template, ordinal, or evidence identifier to it.
            completion = complete_fn(checkpoint, request.prompt, request.signature)
            record_bytes = _canonical_json(_completion_record(completion))
            raw_seals.append((record_bytes, sha256(record_bytes).hexdigest()))

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
    return tuple(sealed)


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


def score_sealed_completions(
    protocol: CompositionProtocol,
    sealed: tuple[SealedCompletion, ...],
    *,
    checkpoint_sha256s: Mapping[str, str],
    checkpoint_cell_synapse_sha256s: Mapping[str, str],
) -> ScoredEvaluation:
    """Score labels and semantics only after the caller has sealed all lanes."""
    expected_total = len(protocol.lanes) * (
        len(protocol.train_tasks) + len(protocol.calibration_tasks)
    )
    expected_sequence = tuple(
        (lane.lane, task.cohort, task.task_id)
        for lane in protocol.lanes
        for task in (*protocol.train_tasks, *protocol.calibration_tasks)
    )
    sequence_ok = (
        len(sealed) == expected_total
        and tuple(item.sequence_number for item in sealed) == tuple(range(1, expected_total + 1))
        and tuple((item.lane, item.cohort, item.task_id) for item in sealed) == expected_sequence
    )
    decoded: dict[tuple[str, str], dict[str, object]] = {}
    seal_integrity = sequence_ok
    for item in sealed:
        key = (item.lane, item.task_id)
        if key in decoded:
            seal_integrity = False
            continue
        try:
            decoded[key] = _decode_sealed(item)
        except ValueError:
            seal_integrity = False

    lane_records: dict[str, object] = {}
    a2_train_replay = 0
    a2_train_cases = 0
    a2_calibration_replay = 0
    a2_calibration_cases = 0
    expected_lanes = tuple(lane.lane for lane in protocol.lanes)
    digest_pattern = re.compile(r"[0-9a-f]{64}")
    checkpoint_maps_ok = (
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
    common_integrity = seal_integrity and checkpoint_maps_ok
    a2_decisions: list[tuple[tuple[int, ...], bytes, int]] = []
    a2_train_margins: list[float] = []

    for lane in protocol.lanes:
        train_replay = 0
        train_cases = 0
        calibration_replay = 0
        calibration_cases = 0
        lane_integrity = True
        lane_checkpoint_hashes: set[str] = set()
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
            prompt_ok = (
                record.get("prompt_sha256") == sha256(task.prompt.encode("utf-8")).hexdigest()
            )
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
        lane_records[lane.lane] = {
            "diagnostic_only": lane.diagnostic_only,
            "eligible_lane": lane.eligible,
            "train_replay_tasks": train_replay,
            "train_cases": train_cases,
            "calibration_replay_tasks": calibration_replay,
            "calibration_cases": calibration_cases,
            "integrity_passed": lane_integrity,
        }
        common_integrity &= lane_integrity
        if lane.lane == "A2":
            a2_train_replay = train_replay
            a2_train_cases = train_cases
            a2_calibration_replay = calibration_replay
            a2_calibration_cases = calibration_cases

    collision_groups: dict[tuple[tuple[int, ...], bytes], set[int]] = {}
    for allowed, eligibility, target in a2_decisions:
        collision_groups.setdefault((allowed, eligibility), set()).add(target)
    collision_count = sum(len(targets) > 1 for targets in collision_groups.values())
    minimum_margin = min(a2_train_margins) if a2_train_margins else 0.0
    metrics = A2EligibilityMetrics(
        train_replay_tasks=a2_train_replay,
        train_cases=a2_train_cases,
        calibration_replay_tasks=a2_calibration_replay,
        calibration_cases=a2_calibration_cases,
        common_integrity_gates_passed=common_integrity,
        decision_collision_count=collision_count,
        competitive_train_margin_count=len(a2_train_margins),
        minimum_train_target_margin_pa=minimum_margin,
    )
    eligible = a2_is_eligible("A2", metrics)
    record = {
        "protocol_sha256": protocol.raw_sha256,
        "sealed_completion_count": len(sealed),
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
            "target_margin_distribution": _aggregate(a2_train_margins),
        },
        "diagnostics": {
            "state_separation": _state_separation(protocol, decoded),
            "direct_prompt_trace_retention": _direct_trace_retention(protocol),
            "a0_a1_hypotheses_are_nonveto": True,
        },
    }
    return ScoredEvaluation(record=record, a2_metrics=metrics, a2_eligible=eligible)


def _write_new(path: Path, value: bytes) -> None:
    try:
        with path.open("xb") as handle:
            handle.write(value)
    except FileExistsError as error:
        raise EvidenceExistsError(f"evidence target already exists: {path.name}") from error


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
) -> ScoredEvaluation:
    """Shared implementation; injected dependencies are always marked synthetic."""
    if not synthetic_test_only and (grow_fn is not grow or complete_fn is not complete):
        raise TypeError("production execution requires fixed public grow and complete")
    protocol = load_composition_protocol(protocol_path)
    validate_execution_confirmation(protocol, confirmation)
    plan = build_execution_plan(root, protocol_path)
    validate_source_bundle_confirmation(plan, source_bundle_confirmation)
    runtime_backend = _require_cpu_runtime_backend()
    execution_plan = dict(plan)
    execution_plan["mode"] = "synthetic-test" if synthetic_test_only else "execute"
    execution_plan["material_execution_authorized"] = not synthetic_test_only
    execution_plan["synthetic_test_only"] = synthetic_test_only
    execution_plan["confirmed_protocol_sha256"] = confirmation
    execution_plan["confirmed_source_bundle_sha256"] = source_bundle_confirmation
    execution_plan["runtime_backend"] = runtime_backend
    execution_plan["run_id"] = run_id
    plan_bytes = _canonical_json(execution_plan)
    run_directory = create_evidence_run(evidence_root, run_id, plan_bytes)

    checkpoint_bytes: dict[str, bytes] = {}
    checkpoints: dict[str, object] = {}
    checkpoint_hashes: dict[str, str] = {}
    checkpoint_cell_synapse_hashes: dict[str, str] = {}
    checkpoint_accounting: dict[str, dict[str, int]] = {}
    native_library_identity_before: dict[str, int | str] | None = None
    native_library_identity_after: dict[str, int | str] | None = None
    try:
        if not synthetic_test_only:
            native_library_identity_before = native_runtime.loaded_library_identity().to_record()
            _write_new(
                run_directory / "005-native-library-before.json",
                _canonical_json(
                    {
                        "event": "native-library-identity-before-model",
                        "identity": native_library_identity_before,
                    }
                ),
            )
        examples = tuple(protocol.training_examples())
        for lane in protocol.lanes:
            checkpoint = grow_fn(Layer1Protocol.for_lane(lane.lane), examples)
            if not synthetic_test_only and type(checkpoint) is not FrozenCheckpoint:
                raise TypeError("production growth must return an exact FrozenCheckpoint")
            if not synthetic_test_only:
                current_native_identity = native_runtime.loaded_library_identity().to_record()
                if current_native_identity != native_library_identity_before:
                    raise RuntimeError("loaded native CPU library changed between lanes")
            if getattr(checkpoint, "lane", None) != lane.lane:
                raise ValueError("grown checkpoint lane differs from requested lane")
            checkpoint_accounting[lane.lane] = _validate_checkpoint_accounting(
                protocol,
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
            checkpoint_hashes[lane.lane] = raw_sha256
            cell_synapse_sha256 = getattr(checkpoint, "cell_synapse_sha256", None)
            if (
                not isinstance(cell_synapse_sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", cell_synapse_sha256) is None
            ):
                raise ValueError("checkpoint cell/synapse SHA-256 is invalid")
            checkpoint_cell_synapse_hashes[lane.lane] = cell_synapse_sha256
            checkpoints[lane.lane] = checkpoint
            _write_new(run_directory / f"010-checkpoint-{lane.lane}.json", raw)

        # This phase completes every lane and cohort. No task label or semantic
        # case is inspected until the returned byte seals are all present.
        sealed = seal_all_lane_completions(
            protocol,
            checkpoints,
            complete_fn=complete_fn,
        )
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
            run_directory / "020-sealed-completions.json",
            _canonical_json(sealed_record),
        )

        for lane, checkpoint in checkpoints.items():
            if checkpoint.to_bytes() != checkpoint_bytes[lane]:  # type: ignore[attr-defined]
                raise RuntimeError("completion changed checkpoint raw bytes")
            if sha256(checkpoint_bytes[lane]).hexdigest() != checkpoint_hashes[lane]:
                raise RuntimeError("checkpoint raw-byte digest changed")
        evaluation = score_sealed_completions(
            protocol,
            sealed,
            checkpoint_sha256s=checkpoint_hashes,
            checkpoint_cell_synapse_sha256s=checkpoint_cell_synapse_hashes,
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
            after_plan["protocol_sha256"] != plan["protocol_sha256"]
            or after_plan["source_bundle_sha256"] != plan["source_bundle_sha256"]
        ):
            raise RuntimeError("protocol or source bundle changed during execution")
        if not synthetic_test_only:
            native_library_identity_after = native_runtime.loaded_library_identity().to_record()
            if native_library_identity_after != native_library_identity_before:
                raise RuntimeError("loaded native CPU library bytes changed during execution")
            _write_new(
                run_directory / "025-native-library-after.json",
                _canonical_json(
                    {
                        "event": "native-library-identity-before-publish",
                        "identity": native_library_identity_after,
                    }
                ),
            )
        evaluation.record["source_bundle_sha256"] = plan["source_bundle_sha256"]
        evaluation.record["source_bundle_unchanged"] = True
        evaluation.record["runtime_backend"] = runtime_backend
        evaluation.record["native_library_identity_before"] = native_library_identity_before
        evaluation.record["native_library_identity_after"] = native_library_identity_after
        evaluation.record["checkpoint_raw_bytes_unchanged"] = True
        evaluation.record["checkpoint_accounting"] = checkpoint_accounting
        evaluation.record["observation_and_strengthen_accounting_passed"] = True
        evaluation.record["exclusive_evidence_creation"] = True
        _write_new(
            run_directory / "030-evaluation.json",
            _canonical_json(evaluation.record),
        )
        summary = {
            "event": "terminal-summary",
            "protocol_sha256": protocol.raw_sha256,
            "source_bundle_sha256": plan["source_bundle_sha256"],
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
            run_directory / "999-terminal-summary.json",
            _canonical_json(summary),
        )
        return evaluation
    except BaseException as error:
        _terminal_failure(run_directory, error)
        raise


def execute_protocol(
    root: Path,
    protocol_path: Path,
    *,
    evidence_root: Path,
    run_id: str,
    confirmation: str | None,
    source_bundle_confirmation: str | None,
) -> ScoredEvaluation:
    """Execute the exact public learner and decoder once into exclusive evidence."""
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
    )
