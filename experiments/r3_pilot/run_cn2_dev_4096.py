"""Run the compact CN2-DEV-4096 public-TRAIN development gate.

This is a replace-not-retry successor to calibration-v3 scoring.  It checks
factorized per-column identity, durable exact replay in READ_ONLY mode, and a
developmental (non-C1-authorizing) full-schedule cost projection.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import math
import platform
import statistics
import struct
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import humanbrain.native_temporal_runtime as runtime_adapter  # noqa: E402, I001
from experiments.r3_pilot import calibration_v3_protocol as definition_protocol  # noqa: E402
from experiments.r3_pilot.calibration_v3_definitions import (  # noqa: E402
    build_r3_f0_definition,
)
from experiments.r3_pilot.r3_f0_train_input import (  # noqa: E402
    load_verified_train_input,
)
from humanbrain.native import loaded_library_identity  # noqa: E402
from humanbrain.native_temporal_runtime import (  # noqa: E402
    NativeTemporalCircuit,
    NativeTemporalCircuitDefinition,
    TemporalCheckpointPublication,
    TemporalCircuitStep,
    TemporalStepMode,
)


ARTIFACT_ID = "cn2-dev-4096-v1"
OUTPUT_PATH = Path(__file__).with_name("cn2_dev_4096.json")
PREFIX_TOKENS = 4_096
ACTIVE_COLUMNS = 11
ARM = (16, 18)
CONSTRUCTION_SEED = 17
FULL_TRAIN_TOKENS = 2_117_412
FULL_TRAIN_TRANSITIONS = 2_117_363
FULL_TRAIN_BOOKS = 49
FULL_TRAIN_STREAM_SHA256 = "f5d2ef9a096e1bb184af925d0523e3a3b80c5bdf7e821b9fca9f65c66059e7b5"
PREFIX_SHA256_LE_I32 = "a39faff31eb8be4c13f4b8cd5e4b070009eeca575bde44ad552b8bf8dcbef26f"
WORST_CASE_SEGMENT_CAPACITY = FULL_TRAIN_TRANSITIONS * ACTIVE_COLUMNS
TOTAL_C1_SCHEDULE_STEPS = 33_135_753
DEVELOPMENT_PROJECTION_CAP_SECONDS = int(13.5 * 60 * 60)
GATE_WALL_CAP_NS = 300_000_000_000

GATE_CONTRACT: dict[str, object] = {
    "artifact_id": ARTIFACT_ID,
    "arm": list(ARM),
    "construction_seed": CONSTRUCTION_SEED,
    "input": {
        "partition": "public TRAIN only",
        "prefix_tokens": PREFIX_TOKENS,
        "prefix_sha256_le_i32": PREFIX_SHA256_LE_I32,
        "train_stream_sha256": FULL_TRAIN_STREAM_SHA256,
    },
    "identity": {
        "novel_complete_key": (
            "allocation is the native actual 0..11 count; complete ordered tuple "
            "is globally new; every previously bound exact-source output column "
            "retains its owner"
        ),
        "repeated_complete_key": "allocation 0 and exact stored ordered tuple",
        "terminal_segments": "sum of native allocated_segments",
    },
    "evaluation": (
        "reopen durable LEARN checkpoint; replay identical prefix READ_ONLY; "
        "require exact tuple trace, zero allocation/punishment, and byte-identical checkpoint"
    ),
    "cost": {
        "active_wall_cap_ns": GATE_WALL_CAP_NS,
        "development_projection_cap_seconds": DEVELOPMENT_PROJECTION_CAP_SECONDS,
        "schedule_steps": TOTAL_C1_SCHEDULE_STEPS,
        "segment_capacity": WORST_CASE_SEGMENT_CAPACITY,
    },
    "claim_boundary": {
        "authorizes_c1": False,
        "capability_evidence": False,
        "held_out_data_accessed": False,
        "threshold_selection": False,
        "transfer_evidence": False,
    },
}


class GateStop(RuntimeError):
    """One decisive development-gate stop."""

    def __init__(self, kind: str, step: int, details: dict[str, object]) -> None:
        super().__init__(kind)
        self.kind = kind
        self.step = step
        self.details = details


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _canonical_lf_sha256(path: Path) -> str:
    raw = path.read_bytes().replace(b"\r\n", b"\n")
    if b"\r" in raw:
        raise RuntimeError(f"source contains a lone carriage return: {path}")
    return hashlib.sha256(raw).hexdigest()


def _source_binding() -> dict[str, object]:
    fixed = (
        Path(__file__).resolve(),
        REPO_ROOT / "experiments/r3_pilot/calibration_v3_definitions.py",
        REPO_ROOT / "experiments/r3_pilot/calibration_v3_protocol.py",
        REPO_ROOT / "experiments/r3_pilot/r3_f0_train_input.py",
        REPO_ROOT / "experiments/r3_pilot/r3_f0_train_input_v1.json",
        REPO_ROOT / "src/humanbrain/cortex.py",
        REPO_ROOT / "src/humanbrain/native.py",
        REPO_ROOT / "src/humanbrain/native_temporal_runtime.py",
        REPO_ROOT / "src/humanbrain/sdr_union.py",
        REPO_ROOT / "src/humanbrain/_native_src/.cargo/config.toml",
        REPO_ROOT / "src/humanbrain/_native_src/Cargo.lock",
        REPO_ROOT / "src/humanbrain/_native_src/Cargo.toml",
    )
    native_sources = tuple(sorted((REPO_ROOT / "src/humanbrain/_native_src/src").rglob("*.rs")))
    paths = (*fixed, *native_sources)
    if not native_sources or any(not path.is_file() for path in paths):
        raise RuntimeError("CN2 source-binding closure is incomplete")
    files = {path.relative_to(REPO_ROOT).as_posix(): _canonical_lf_sha256(path) for path in paths}
    source_sha256 = hashlib.sha256(
        _canonical_json_bytes({"domain": "humanbrain-cn2-dev-source-v1", "files": files})
    ).hexdigest()
    protocol_sha256 = hashlib.sha256(_canonical_json_bytes(GATE_CONTRACT)).hexdigest()
    return {
        "canonical_byte_rule": "SHA-256 after CRLF-to-LF normalization; lone CR forbidden",
        "files": files,
        "protocol_identity_sha256": protocol_sha256,
        "source_bundle_identity_sha256": source_sha256,
    }


@contextlib.contextmanager
def _development_authority(binding: dict[str, object]) -> Iterator[None]:
    """Bind native operations to this development gate's live source closure."""

    original = runtime_adapter._fresh_authority
    authority = runtime_adapter._FreshAuthority(
        bytes.fromhex(str(binding["protocol_identity_sha256"])),
        bytes.fromhex(str(binding["source_bundle_identity_sha256"])),
    )
    runtime_adapter._fresh_authority = lambda: authority
    try:
        yield
    finally:
        runtime_adapter._fresh_authority = original


def _nearest_rank(values: Sequence[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def _distribution(values: Sequence[int]) -> dict[str, float | int]:
    if not values:
        raise ValueError("distribution requires at least one value")
    return {
        "count": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "p95": _nearest_rank(values, 0.95),
        "p99": _nearest_rank(values, 0.99),
        "max": max(values),
        "sum": sum(values),
    }


def _winner_tuple(step: TemporalCircuitStep, *, expected_event_count: int) -> tuple[int, ...]:
    if step.event_count != expected_event_count:
        raise GateStop(
            "EVENT_COUNT_DRIFT",
            expected_event_count - 1,
            {"expected": expected_event_count, "observed": step.event_count},
        )
    winners = tuple(spike.cell_id for spike in step.winner_spikes)
    if len(winners) != ACTIVE_COLUMNS or len(set(winners)) != ACTIVE_COLUMNS:
        raise GateStop(
            "WINNER_TUPLE_SHAPE_DRIFT",
            expected_event_count - 1,
            {"winner_count": len(winners), "unique_winners": len(set(winners))},
        )
    if not (
        0 <= step.active_segments <= step.matching_segments <= step.candidate_segments
        and step.candidate_segments <= step.postings_visited
        and 0 <= step.predicted_columns <= ACTIVE_COLUMNS
        and 0 <= step.allocated_segments <= ACTIVE_COLUMNS
        and step.punished_segments >= 0
    ):
        raise GateStop(
            "NATIVE_COUNTER_CONTRACT_DRIFT",
            expected_event_count - 1,
            {
                "active": step.active_segments,
                "matching": step.matching_segments,
                "candidate": step.candidate_segments,
                "postings": step.postings_visited,
                "predicted_columns": step.predicted_columns,
                "allocated": step.allocated_segments,
                "punished": step.punished_segments,
            },
        )
    return winners


def _trace_sha256(trace: Sequence[tuple[int, ...]]) -> str:
    digest = hashlib.sha256()
    for winners in trace:
        digest.update(struct.pack("<11I", *winners))
    return digest.hexdigest()


class FactorizedIdentityLedger:
    """Small observational ledger for the corrected per-column contract."""

    def __init__(self) -> None:
        self.complete_keys: dict[tuple[tuple[int, ...], int], tuple[int, ...]] = {}
        self.column_owners: dict[tuple[tuple[int, ...], int], int] = {}
        self.tuple_owners: dict[tuple[int, ...], tuple[str, int]] = {}
        self.previous: tuple[int, ...] | None = None
        self.event_count = 0
        self.trace: list[tuple[int, ...]] = []
        self.novel_keys = 0
        self.repeated_keys = 0
        self.partial_novel_allocations = 0
        self.actual_allocations = 0
        self.allocation_histogram: Counter[int] = Counter()

    def observe(
        self,
        step: TemporalCircuitStep,
        *,
        columns: tuple[int, ...],
        reset_before: bool,
    ) -> tuple[int, ...]:
        if len(columns) != ACTIVE_COLUMNS or len(set(columns)) != ACTIVE_COLUMNS:
            raise ValueError("token columns must contain 11 unique entries")
        self.event_count = 1 if reset_before else self.event_count + 1
        winners = _winner_tuple(step, expected_event_count=self.event_count)
        location = len(self.trace)
        self.trace.append(winners)
        self.actual_allocations += step.allocated_segments
        self.allocation_histogram[step.allocated_segments] += 1

        if reset_before:
            if step.allocated_segments or step.punished_segments:
                raise GateStop(
                    "RESET_MUTATED_LEARNED_STATE",
                    location,
                    {
                        "allocated": step.allocated_segments,
                        "punished": step.punished_segments,
                    },
                )
            prior = self.tuple_owners.get(winners)
            seed_identity = ("reset_seed", step.token_id)
            if prior is not None and prior != seed_identity:
                raise GateStop(
                    "RESET_REUSED_LEARNED_COMPLETE_TUPLE",
                    location,
                    {"prior_kind": prior[0], "prior_token": prior[1]},
                )
            self.tuple_owners[winners] = seed_identity
            self.previous = winners
            return winners

        if self.previous is None:
            raise GateStop("TRANSITION_WITHOUT_SOURCE", location, {})
        source = self.previous
        key = (source, step.token_id)
        stored = self.complete_keys.get(key)
        if stored is not None:
            if step.allocated_segments != 0 or winners != stored:
                raise GateStop(
                    "REPEATED_COMPLETE_KEY_DRIFT",
                    location,
                    {
                        "allocated": step.allocated_segments,
                        "tuple_equal": winners == stored,
                    },
                )
            self.repeated_keys += 1
        else:
            prior = self.tuple_owners.get(winners)
            if prior is not None:
                raise GateStop(
                    "NOVEL_KEY_REUSED_COMPLETE_TUPLE",
                    location,
                    {"prior_kind": prior[0], "prior_token": prior[1]},
                )
            bound = 0
            for column, winner in zip(columns, winners, strict=True):
                binding_key = (source, column)
                owner = self.column_owners.get(binding_key)
                if owner is not None:
                    bound += 1
                    if winner != owner:
                        raise GateStop(
                            "EXACT_SOURCE_COLUMN_OWNER_DRIFT",
                            location,
                            {"column": column, "expected": owner, "observed": winner},
                        )
            if step.allocated_segments > ACTIVE_COLUMNS - bound:
                raise GateStop(
                    "ALLOCATION_EXCEEDS_UNBOUND_COLUMNS",
                    location,
                    {
                        "allocated": step.allocated_segments,
                        "known_bindings": bound,
                        "unbound_columns": ACTIVE_COLUMNS - bound,
                    },
                )
            self.complete_keys[key] = winners
            self.tuple_owners[winners] = ("complete_key", step.token_id)
            for column, winner in zip(columns, winners, strict=True):
                self.column_owners.setdefault((source, column), winner)
            self.novel_keys += 1
            self.partial_novel_allocations += int(step.allocated_segments < ACTIVE_COLUMNS)
        self.previous = winners
        return winners

    def summary(self) -> dict[str, object]:
        return {
            "observed_steps": len(self.trace),
            "complete_key_count": len(self.complete_keys),
            "column_binding_count": len(self.column_owners),
            "novel_complete_keys": self.novel_keys,
            "repeated_complete_keys": self.repeated_keys,
            "partial_novel_allocation_events": self.partial_novel_allocations,
            "actual_allocated_segments": self.actual_allocations,
            "allocation_histogram": {
                str(key): value for key, value in sorted(self.allocation_histogram.items())
            },
            "winner_trace_sha256_le_u32": _trace_sha256(self.trace),
        }


class CostBands:
    def __init__(self) -> None:
        self.samples: dict[str, list[tuple[int, int]]] = {
            "1k_to_8k": [],
            "8k_to_32k": [],
            "32k_plus": [],
        }

    def observe(self, live_segments: int, latency_ns: int) -> None:
        if live_segments < 1_000:
            return
        if live_segments < 8_000:
            band = "1k_to_8k"
        elif live_segments < 32_000:
            band = "8k_to_32k"
        else:
            band = "32k_plus"
        self.samples[band].append((live_segments, latency_ns))

    def points(self) -> tuple[list[tuple[float, float]], dict[str, object]]:
        points: list[tuple[float, float]] = []
        records: dict[str, object] = {}
        for band, samples in self.samples.items():
            if not samples:
                raise GateStop("COST_BAND_EMPTY", PREFIX_TOKENS, {"band": band})
            live = [sample[0] for sample in samples]
            latency = [sample[1] for sample in samples]
            live_median = float(statistics.median(live))
            latency_distribution = _distribution(latency)
            points.append((live_median, float(latency_distribution["p95"])))
            records[band] = {
                "live_segments": _distribution(live),
                "latency_ns": latency_distribution,
            }
        return points, records


def _log_log_fit(points: Sequence[tuple[float, float]]) -> dict[str, float]:
    if len(points) < 3 or any(x <= 0 or y <= 0 for x, y in points):
        raise ValueError("cost fit requires three positive points")
    xs = [math.log(x) for x, _ in points]
    ys = [math.log(y) for _, y in points]
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    raw_slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True)) / denominator
    slope = max(0.0, raw_slope)
    intercept = y_mean - slope * x_mean
    fitted = [intercept + slope * x for x in xs]
    residual = sum((y - estimate) ** 2 for y, estimate in zip(ys, fitted, strict=True))
    total = sum((y - y_mean) ** 2 for y in ys)
    r_squared = 1.0 if total == 0.0 and residual == 0.0 else 1.0 - residual / total
    return {
        "raw_exponent": raw_slope,
        "nonnegative_exponent": slope,
        "log_intercept": intercept,
        "r_squared": r_squared,
    }


def _checkpoint_publication(publication: TemporalCheckpointPublication) -> dict[str, object]:
    return {
        "outcome": publication.outcome.name,
        "semantically_resumable": publication.semantically_resumable,
        "checkpoint_bytes": publication.checkpoint_bytes,
        "encoded_bytes": publication.encoded_bytes,
        "temp_cleanup": publication.temp_cleanup.name,
    }


def _path_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _definition() -> tuple[dict[str, Any], NativeTemporalCircuitDefinition]:
    protocol = definition_protocol.build_protocol()
    definition_protocol.validate_protocol(protocol, require_runnable=False)
    production = build_r3_f0_definition(protocol, ARM, CONSTRUCTION_SEED)
    config = replace(production.config, segment_capacity=WORST_CASE_SEGMENT_CAPACITY)
    return protocol, NativeTemporalCircuitDefinition(
        config,
        production.token_codes,
        production.threshold_offsets_mv,
    )


def _load_prefix() -> tuple[object, object, np.ndarray]:
    verified = load_verified_train_input()
    books = tuple(verified.iter_books())
    if (
        verified.token_count != FULL_TRAIN_TOKENS
        or verified.train_stream_sha256 != FULL_TRAIN_STREAM_SHA256
        or len(books) != FULL_TRAIN_BOOKS
        or books[0].short_name != "Genesis"
        or books[0].token_count < PREFIX_TOKENS
    ):
        raise RuntimeError("canonical public TRAIN authority drifted")
    prefix = np.frombuffer(books[0].token_bytes_le_i32, dtype="<i4")[:PREFIX_TOKENS]
    if hashlib.sha256(prefix.tobytes()).hexdigest() != PREFIX_SHA256_LE_I32:
        raise RuntimeError("canonical Genesis prefix identity drifted")
    return verified, books[0], prefix


def _run_gate(binding: dict[str, object], scratch: Path) -> dict[str, object]:
    learned_path = scratch / "learned.checkpoint"
    readonly_path = scratch / "readonly.checkpoint"
    active_started_ns = time.perf_counter_ns()
    verified, first_book, prefix = _load_prefix()
    protocol, definition = _definition()
    token_columns = dict(definition.token_codes)
    if any(int(token) not in token_columns for token in prefix):
        raise RuntimeError("public TRAIN prefix contains an unknown production token")

    ledger = FactorizedIdentityLedger()
    learn_latencies: list[int] = []
    readonly_latencies: list[int] = []
    cost_bands = CostBands()
    with _development_authority(binding):
        with NativeTemporalCircuit.fresh(definition) as circuit:
            gc_enabled = gc.isenabled()
            gc.disable()
            try:
                for index, raw_token in enumerate(prefix):
                    token = int(raw_token)
                    live_before = ledger.actual_allocations
                    started_ns = time.perf_counter_ns()
                    step = circuit.advance_block(
                        [token], mode=TemporalStepMode.LEARN, reset_before=index == 0
                    )[0]
                    latency_ns = time.perf_counter_ns() - started_ns
                    ledger.observe(
                        step,
                        columns=token_columns[token],
                        reset_before=index == 0,
                    )
                    learn_latencies.append(latency_ns)
                    cost_bands.observe(live_before, latency_ns)
                    if (
                        index % 256 == 255
                        and time.perf_counter_ns() - active_started_ns > GATE_WALL_CAP_NS
                    ):
                        raise GateStop(
                            "ACTIVE_WALL_CAP",
                            index,
                            {"cap_ns": GATE_WALL_CAP_NS},
                        )
            finally:
                if gc_enabled:
                    gc.enable()
            learned_identity = circuit.checkpoint_identity()
            if learned_identity.segment_count != ledger.actual_allocations:
                raise GateStop(
                    "TERMINAL_SEGMENT_ACCOUNTING_DRIFT",
                    PREFIX_TOKENS,
                    {
                        "native": learned_identity.segment_count,
                        "actual_allocations": ledger.actual_allocations,
                    },
                )
            learned_publication = circuit.publish_checkpoint(learned_path)
            if not learned_publication.semantically_resumable:
                raise GateStop(
                    "LEARN_CHECKPOINT_NOT_DURABLE",
                    PREFIX_TOKENS,
                    {"outcome": learned_publication.outcome.name},
                )

        with NativeTemporalCircuit.from_checkpoint_path(definition, learned_path) as replay:
            reopened_identity = replay.checkpoint_identity()
            if reopened_identity != learned_identity:
                raise GateStop("CHECKPOINT_IMPORT_IDENTITY_DRIFT", PREFIX_TOKENS, {})
            gc_enabled = gc.isenabled()
            gc.disable()
            try:
                for index, (raw_token, expected) in enumerate(
                    zip(prefix, ledger.trace, strict=True)
                ):
                    token = int(raw_token)
                    started_ns = time.perf_counter_ns()
                    step = replay.advance_block(
                        [token], mode=TemporalStepMode.READ_ONLY, reset_before=index == 0
                    )[0]
                    readonly_latencies.append(time.perf_counter_ns() - started_ns)
                    winners = _winner_tuple(step, expected_event_count=index + 1)
                    if (
                        winners != expected
                        or step.allocated_segments != 0
                        or step.punished_segments != 0
                    ):
                        raise GateStop(
                            "READ_ONLY_REPLAY_DRIFT",
                            index,
                            {
                                "tuple_equal": winners == expected,
                                "allocated": step.allocated_segments,
                                "punished": step.punished_segments,
                            },
                        )
                    if (
                        index % 256 == 255
                        and time.perf_counter_ns() - active_started_ns > GATE_WALL_CAP_NS
                    ):
                        raise GateStop(
                            "ACTIVE_WALL_CAP",
                            PREFIX_TOKENS + index,
                            {"cap_ns": GATE_WALL_CAP_NS},
                        )
            finally:
                if gc_enabled:
                    gc.enable()
            readonly_identity = replay.checkpoint_identity()
            if readonly_identity != learned_identity:
                raise GateStop("READ_ONLY_CHECKPOINT_IDENTITY_DRIFT", PREFIX_TOKENS * 2, {})
            readonly_publication = replay.publish_checkpoint(readonly_path)
            if not readonly_publication.semantically_resumable:
                raise GateStop(
                    "READ_ONLY_CHECKPOINT_NOT_DURABLE",
                    PREFIX_TOKENS * 2,
                    {"outcome": readonly_publication.outcome.name},
                )
        native_identity = loaded_library_identity()

    learned_sha256 = _path_sha256(learned_path)
    readonly_sha256 = _path_sha256(readonly_path)
    if (
        learned_sha256 != readonly_sha256
        or learned_path.stat().st_size != readonly_path.stat().st_size
    ):
        raise GateStop(
            "READ_ONLY_CHECKPOINT_BYTES_DRIFT",
            PREFIX_TOKENS * 2,
            {
                "learned_sha256": learned_sha256,
                "readonly_sha256": readonly_sha256,
            },
        )
    active_wall_ns = time.perf_counter_ns() - active_started_ns
    if active_wall_ns > GATE_WALL_CAP_NS:
        raise GateStop(
            "ACTIVE_WALL_CAP",
            PREFIX_TOKENS * 2,
            {"cap_ns": GATE_WALL_CAP_NS, "observed_ns": active_wall_ns},
        )

    points, band_records = cost_bands.points()
    fit = _log_log_fit(points)
    exponent = fit["nonnegative_exponent"]
    learn_projected_p95_ns = math.exp(
        fit["log_intercept"] + exponent * math.log(WORST_CASE_SEGMENT_CAPACITY)
    )
    readonly_p95_ns = float(_distribution(readonly_latencies)["p95"])
    final_segments = max(1, learned_identity.segment_count)
    readonly_projected_p95_ns = (
        readonly_p95_ns * (WORST_CASE_SEGMENT_CAPACITY / final_segments) ** exponent
    )
    projected_step_ns = max(learn_projected_p95_ns, readonly_projected_p95_ns)
    measured_step_ns = sum(learn_latencies) + sum(readonly_latencies)
    fixed_gate_ns = max(0, active_wall_ns - measured_step_ns)
    projected_total_ns = projected_step_ns * TOTAL_C1_SCHEDULE_STEPS + 3 * fixed_gate_ns
    projected_hours = projected_total_ns / 3_600_000_000_000
    status = (
        "PASS_CN2_DEV_4096"
        if projected_hours < DEVELOPMENT_PROJECTION_CAP_SECONDS / 3_600
        else "STOP_CN2_DEV_COST_PROJECTION"
    )

    return {
        "schema_version": 1,
        "artifact_id": ARTIFACT_ID,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": status,
        "claim_boundary": GATE_CONTRACT["claim_boundary"],
        "source_binding": binding,
        "input": {
            "partition": "public TRAIN",
            "candidate_id": verified.candidate_id,
            "train_input_authority_sha256": verified.train_input_authority_sha256,
            "train_stream_sha256": verified.train_stream_sha256,
            "full_train_token_count": verified.token_count,
            "train_books_materialized": FULL_TRAIN_BOOKS,
            "held_out_books_materialized": 0,
            "prefix_book": first_book.short_name,
            "prefix_token_count": PREFIX_TOKENS,
            "prefix_sha256_le_i32": PREFIX_SHA256_LE_I32,
            "prefix_unique_token_ids": int(np.unique(prefix).size),
        },
        "definition": {
            "arm": list(ARM),
            "construction_seed": CONSTRUCTION_SEED,
            "definition_protocol_sha256": definition_protocol.protocol_sha256(),
            "definition_protocol_runnable": protocol["source_binding"]["runnable"],
            "circuit_identity_sha256": learned_identity.circuit_identity_sha256,
            "segment_capacity": definition.config.segment_capacity,
            "segment_capacity_rule": "2,117,363 TRAIN transitions * 11 columns",
            "column_count": definition.config.column_count,
            "token_code_count": len(definition.token_codes),
        },
        "learn": {
            "mode": "LEARN",
            "identity": ledger.summary(),
            "latency_ns": _distribution(learn_latencies),
            "terminal_checkpoint_identity": asdict(learned_identity),
        },
        "read_only": {
            "mode": "READ_ONLY",
            "steps": PREFIX_TOKENS,
            "exact_winner_trace": True,
            "zero_allocations": True,
            "zero_punishments": True,
            "latency_ns": _distribution(readonly_latencies),
            "terminal_identity_exact": readonly_identity == learned_identity,
            "checkpoint_bytes_exact": learned_sha256 == readonly_sha256,
        },
        "durability": {
            "learned_publication": _checkpoint_publication(learned_publication),
            "readonly_publication": _checkpoint_publication(readonly_publication),
            "checkpoint_bytes": learned_path.stat().st_size,
            "checkpoint_sha256": learned_sha256,
            "reopened_identity_exact": reopened_identity == learned_identity,
        },
        "developmental_cost_projection": {
            "authorizes_c1": False,
            "classification": "developmental extrapolation; not C1 launch evidence",
            "timer": "time.perf_counter_ns",
            "active_wall_ns": active_wall_ns,
            "active_wall_cap_ns": GATE_WALL_CAP_NS,
            "measured_step_ns": measured_step_ns,
            "measured_fixed_gate_ns": fixed_gate_ns,
            "bands": band_records,
            "log_log_p95_fit": fit,
            "worst_case_live_segments": WORST_CASE_SEGMENT_CAPACITY,
            "learn_projected_p95_ns": learn_projected_p95_ns,
            "readonly_projected_p95_ns": readonly_projected_p95_ns,
            "conservative_projected_step_ns": projected_step_ns,
            "schedule_steps": TOTAL_C1_SCHEDULE_STEPS,
            "projected_total_hours": projected_hours,
            "development_cap_hours": DEVELOPMENT_PROJECTION_CAP_SECONDS / 3_600,
            "required_steps_per_second_at_13_5h": (
                TOTAL_C1_SCHEDULE_STEPS / DEVELOPMENT_PROJECTION_CAP_SECONDS
            ),
            "passes_development_cap": projected_hours < 13.5,
            "model_limit": (
                "three-band 4,096-token extrapolation; later-book growth and the "
                "protected evaluation execution are unmeasured"
            ),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "native": native_identity.to_record(),
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite existing CN2 artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    binding = _source_binding()
    try:
        with tempfile.TemporaryDirectory(prefix="cn2-dev-4096-", dir=output.parent) as raw:
            payload = _run_gate(binding, Path(raw))
    except GateStop as exc:
        payload = {
            "schema_version": 1,
            "artifact_id": ARTIFACT_ID,
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "status": f"STOP_CN2_DEV_{exc.kind}",
            "claim_boundary": GATE_CONTRACT["claim_boundary"],
            "source_binding": binding,
            "stop": {"kind": exc.kind, "step": exc.step, "details": exc.details},
        }
    except Exception as exc:
        payload = {
            "schema_version": 1,
            "artifact_id": ARTIFACT_ID,
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "status": "STOP_CN2_DEV_INFRASTRUCTURE",
            "claim_boundary": GATE_CONTRACT["claim_boundary"],
            "source_binding": binding,
            "stop": {"error_type": type(exc).__name__, "error_message": str(exc)},
        }
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(f"{payload['status']}: {output}", flush=True)
    return 0 if payload["status"] == "PASS_CN2_DEV_4096" else 2


if __name__ == "__main__":
    raise SystemExit(main())
