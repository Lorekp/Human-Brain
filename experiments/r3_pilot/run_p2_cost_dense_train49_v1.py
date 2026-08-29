"""Direct full-density public-TRAIN cost gate for Path-2.

Build one real 49-book TRAIN checkpoint with actual allocation accounting only,
then time independent full-density Genesis LEARN and READ_ONLY clones plus
checkpoint I/O.  This runner does not rescore identity, inspect held-out data,
or select thresholds.  It requires an explicit isolated-host confirmation.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import tempfile
import time
from collections.abc import Iterator, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import humanbrain.native_temporal_runtime as runtime_adapter  # noqa: E402, I001
from experiments.r3_pilot import run_cn2_dev_4096 as cn2  # noqa: E402
from experiments.r3_pilot.r3_f0_train_input import (  # noqa: E402
    load_verified_train_input,
)
from humanbrain.native import loaded_library_identity  # noqa: E402
from humanbrain.native_temporal_runtime import (  # noqa: E402
    NativeTemporalCircuit,
    TemporalCheckpointPublication,
    TemporalStepMode,
)


ARTIFACT_ID = "P2-COST-DENSE-TRAIN49-v1"
OUTPUT_PATH = Path(__file__).with_name("p2_cost_dense_train49_v1.json")
BASE_CHECKPOINT_PATH = Path(__file__).with_name("p2_cost_dense_train49_v1.base.checkpoint")
CN2_INTERPRETATION_PATH = Path(__file__).with_name("cn2_dev_4096.interpretation.json")
BLOCK_TOKENS = 256
MAX_BUILD_CHUNK_BOOKS = 2
PARTIAL_MANIFEST_SCHEMA_VERSION = 1
GENESIS_TOKENS = 122_253
HELD_OUT_SCHEDULE_TOKENS = 458_191
PER_ARM_SCHEDULE_STEPS = 11_045_251
DENSE_STEPS_AFTER_EPOCH0_PER_ARM = PER_ARM_SCHEDULE_STEPS - cn2.FULL_TRAIN_TOKENS
SERIALIZED_ARMS = 3
CHECKPOINT_EXPORTS_PER_ARM = 5
RESUME_IMPORTS_PER_ARM = 1
RUN_WALL_CAP_NS = 6 * 60 * 60 * 1_000_000_000
MIN_FREE_DISK_BYTES = 6 * 1024**3
MIN_REPLAY_CPU_WALL_RATIO = 0.90
MAX_CHECKPOINT_BYTES = 256 + 64 * cn2.WORST_CASE_SEGMENT_CAPACITY + 4 * 12_288 + 2 * 22 * 4
EXPECTED_PEAK_CHECKPOINT_BYTES = 3 * MAX_CHECKPOINT_BYTES

GATE_CONTRACT: dict[str, object] = {
    "artifact_id": ARTIFACT_ID,
    "input": {
        "partition": "public TRAIN only",
        "books": cn2.FULL_TRAIN_BOOKS,
        "tokens": cn2.FULL_TRAIN_TOKENS,
        "train_stream_sha256": cn2.FULL_TRAIN_STREAM_SHA256,
    },
    "definition": {
        "arm": list(cn2.ARM),
        "construction_seed": cn2.CONSTRUCTION_SEED,
        "segment_capacity": cn2.WORST_CASE_SEGMENT_CAPACITY,
        "threshold_choice": False,
    },
    "setup": (
        "resumable exact 49-book TRAIN LEARN pass in one- or two-book chunks; "
        "publish an authenticated generation checkpoint plus atomic cursor manifest, "
        "record only native actual allocations, and require terminal segment_count "
        "to equal their sum"
    ),
    "dense_measurement": (
        "independently import the full checkpoint for Genesis LEARN and READ_ONLY "
        "block replays; time imports, replays, exports, and hashes"
    ),
    "checkpoint_disk": {
        "maximum_bytes_each": MAX_CHECKPOINT_BYTES,
        "simultaneous_files": 3,
        "expected_peak_bytes": EXPECTED_PEAK_CHECKPOINT_BYTES,
    },
    "timing_quality": {
        "clock": "time.process_time_ns / time.perf_counter_ns",
        "minimum_replay_cpu_wall_ratio": MIN_REPLAY_CPU_WALL_RATIO,
        "low_ratio_verdict": "STOP_P2_COST_DENSE_TRAIN49_CONTENTION",
    },
    "claim_boundary": {
        "authorizes_c1": False,
        "capability_evidence": False,
        "held_out_data_accessed": False,
        "identity_rescored": False,
        "threshold_selection": False,
        "transfer_evidence": False,
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--base-checkpoint", type=Path, default=BASE_CHECKPOINT_PATH)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="validate fixed inputs/source/disk and print a no-run summary",
    )
    parser.add_argument(
        "--confirm-no-heavy-process",
        action="store_true",
        help="orchestrator attestation that no competing CPU/native workload is live",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--build-chunk-books",
        type=int,
        metavar="N",
        help=(
            "build and durably publish one resumable public-TRAIN chunk "
            f"(1..{MAX_BUILD_CHUNK_BOOKS} books)"
        ),
    )
    mode.add_argument(
        "--measure-complete-build",
        action="store_true",
        help="measure dense clones from an authenticated complete TRAIN49 build",
    )
    args = parser.parse_args()
    if args.build_chunk_books is not None and not (
        1 <= args.build_chunk_books <= MAX_BUILD_CHUNK_BOOKS
    ):
        parser.error(f"--build-chunk-books must be in 1..{MAX_BUILD_CHUNK_BOOKS}")
    if (
        not args.preflight_only
        and args.build_chunk_books is None
        and not args.measure_complete_build
    ):
        parser.error("choose --build-chunk-books N or --measure-complete-build")
    if not args.preflight_only and not args.confirm_no_heavy_process:
        parser.error("--confirm-no-heavy-process is required for cost evidence")
    return args


def _path_sha256(path: Path) -> tuple[str, int]:
    started_ns = time.perf_counter_ns()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest(), time.perf_counter_ns() - started_ns


def _partial_manifest_path(base_checkpoint: Path) -> Path:
    return base_checkpoint.with_name(f"{base_checkpoint.stem}.partial.json")


def _partial_checkpoint_path(base_checkpoint: Path, next_book_ordinal: int) -> Path:
    return base_checkpoint.with_name(
        f"{base_checkpoint.stem}.partial.book-{next_book_ordinal:02d}.checkpoint"
    )


def _manifest_checkpoint_path(manifest_path: Path, manifest: dict[str, object]) -> Path:
    checkpoint = manifest.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise RuntimeError("partial build manifest lacks checkpoint metadata")
    raw_name = checkpoint.get("file")
    if not isinstance(raw_name, str):
        raise RuntimeError("partial build checkpoint filename is invalid")
    lexical = Path(raw_name)
    if lexical.is_absolute() or lexical.name != raw_name:
        raise RuntimeError("partial build checkpoint must be a sibling filename")
    return manifest_path.parent / lexical


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temp = tempfile.mkstemp(prefix=f"{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(raw_temp)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _build_authority(
    definition: object,
    binding: dict[str, object],
) -> dict[str, object]:
    return {
        "train_stream_sha256": cn2.FULL_TRAIN_STREAM_SHA256,
        "books": cn2.FULL_TRAIN_BOOKS,
        "tokens": cn2.FULL_TRAIN_TOKENS,
        "arm": list(cn2.ARM),
        "construction_seed": cn2.CONSTRUCTION_SEED,
        "segment_capacity": definition.config.segment_capacity,
        "definition_protocol_sha256": cn2.definition_protocol.protocol_sha256(),
        "protocol_identity_sha256": binding["protocol_identity_sha256"],
        "source_bundle_identity_sha256": binding["source_bundle_identity_sha256"],
        "mode": TemporalStepMode.LEARN.name,
        "reset_before_each_book": True,
    }


def _validate_build_manifest(
    manifest_path: Path,
    authority: dict[str, object],
    books: Sequence[object],
) -> dict[str, object] | None:
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("partial build manifest is unreadable") from exc
    if not isinstance(manifest, dict):
        raise RuntimeError("partial build manifest must be a JSON object")
    if (
        manifest.get("schema_version") != PARTIAL_MANIFEST_SCHEMA_VERSION
        or manifest.get("artifact_id") != ARTIFACT_ID
        or manifest.get("authority") != authority
    ):
        raise RuntimeError("partial build manifest authority drifted")
    cursor = manifest.get("cursor")
    rows = manifest.get("books")
    checkpoint = manifest.get("checkpoint")
    if not isinstance(cursor, dict) or not isinstance(checkpoint, dict):
        raise RuntimeError("partial build manifest structure is incomplete")
    if not isinstance(rows, list):
        raise RuntimeError("partial build book ledger is invalid")
    next_ordinal = cursor.get("next_book_ordinal")
    if type(next_ordinal) is not int or not 0 < next_ordinal <= len(books):
        raise RuntimeError("partial build cursor is outside the TRAIN book range")
    if len(rows) != next_ordinal:
        raise RuntimeError("partial build cursor and book ledger disagree")
    expected_tokens = 0
    cumulative_allocations = 0
    for ordinal, (book, row) in enumerate(zip(books[:next_ordinal], rows, strict=True)):
        if not isinstance(row, dict):
            raise RuntimeError("partial build book row is invalid")
        if (
            row.get("ordinal") != ordinal
            or row.get("book") != book.short_name
            or row.get("tokens") != book.token_count
        ):
            raise RuntimeError("partial build book prefix drifted")
        expected_tokens += int(book.token_count)
        allocated = row.get("actual_allocated_segments")
        if type(allocated) is not int or allocated < 0:
            raise RuntimeError("partial build allocation row is invalid")
        cumulative_allocations += allocated
        if row.get("cumulative_actual_allocated_segments") != cumulative_allocations:
            raise RuntimeError("partial build cumulative allocation ledger drifted")
    if cursor.get("token_count") != expected_tokens:
        raise RuntimeError("partial build token cursor drifted")
    if manifest.get("actual_allocated_segments") != cumulative_allocations:
        raise RuntimeError("partial build terminal accounting drifted")
    checkpoint_identity = checkpoint.get("identity")
    if (
        not isinstance(checkpoint_identity, dict)
        or checkpoint_identity.get("segment_count") != cumulative_allocations
    ):
        raise RuntimeError("partial build checkpoint identity drifted")
    checkpoint_path = _manifest_checkpoint_path(manifest_path, manifest)
    if not checkpoint_path.is_file():
        raise RuntimeError("partial build checkpoint is missing")
    expected_bytes = checkpoint.get("bytes")
    if type(expected_bytes) is not int or checkpoint_path.stat().st_size != expected_bytes:
        raise RuntimeError("partial build checkpoint byte count drifted")
    checkpoint_sha256, _hash_ns = _path_sha256(checkpoint_path)
    if checkpoint_sha256 != checkpoint.get("sha256"):
        raise RuntimeError("partial build checkpoint digest drifted")
    expected_status = "BUILD_COMPLETE" if next_ordinal == len(books) else "BUILD_PARTIAL"
    if manifest.get("status") != expected_status:
        raise RuntimeError("partial build status and cursor disagree")
    return manifest


def _authority_binding() -> dict[str, object]:
    inherited = cn2._source_binding()
    runner = Path(__file__).resolve()
    files = dict(inherited["files"])
    files[runner.relative_to(REPO_ROOT).as_posix()] = cn2._canonical_lf_sha256(runner)
    source_sha256 = hashlib.sha256(
        cn2._canonical_json_bytes(
            {"domain": "humanbrain-p2-cost-dense-train49-source-v1", "files": files}
        )
    ).hexdigest()
    protocol_sha256 = hashlib.sha256(cn2._canonical_json_bytes(GATE_CONTRACT)).hexdigest()
    return {
        "canonical_byte_rule": inherited["canonical_byte_rule"],
        "files": files,
        "protocol_identity_sha256": protocol_sha256,
        "source_bundle_identity_sha256": source_sha256,
    }


@contextlib.contextmanager
def _cost_authority(binding: dict[str, object]) -> Iterator[None]:
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


def _preflight(output: Path, checkpoint: Path) -> tuple[dict[str, object], dict[str, object]]:
    if output.exists():
        raise RuntimeError("dense cost output already exists; refusing overwrite")
    if checkpoint.exists():
        raise RuntimeError("unmanaged monolithic base checkpoint exists; refusing overwrite")
    if not CN2_INTERPRETATION_PATH.is_file():
        raise RuntimeError("CN2 timing-contamination interpretation is missing")
    interpretation = json.loads(CN2_INTERPRETATION_PATH.read_bytes())
    if (
        interpretation["classification"] != "STOP_COST_TIMING_CONTAMINATED"
        or interpretation["cost_interpretation"]["replacement_measurement"] != ARTIFACT_ID
        or interpretation["identity_interpretation"]["verdict"]
        != "PASS_CN2_DEV_IDENTITY_DURABILITY"
    ):
        raise RuntimeError("CN2 timing-contamination interpretation drifted")
    free = shutil.disk_usage(checkpoint.parent).free
    if free < MIN_FREE_DISK_BYTES:
        raise RuntimeError(
            f"dense checkpoint run requires at least {MIN_FREE_DISK_BYTES} free bytes"
        )
    binding = _authority_binding()
    _protocol, definition = cn2._definition()
    authority = _build_authority(definition, binding)
    manifest_path = _partial_manifest_path(checkpoint)
    partial: dict[str, object] | None = None
    if manifest_path.exists():
        _verified, books, _token_arrays = _load_train()
        partial = _validate_build_manifest(manifest_path, authority, books)
    active_checkpoint = (
        None if partial is None else _manifest_checkpoint_path(manifest_path, partial)
    )
    return interpretation, {
        "status": "PREFLIGHT_READY_NOT_RUN",
        "artifact_id": ARTIFACT_ID,
        "output": str(output),
        "base_checkpoint": str(checkpoint),
        "partial_manifest": str(manifest_path),
        "build_status": "BUILD_NOT_STARTED" if partial is None else partial["status"],
        "next_book_ordinal": 0 if partial is None else partial["cursor"]["next_book_ordinal"],
        "active_partial_checkpoint": (
            None if active_checkpoint is None else str(active_checkpoint)
        ),
        "outputs_absent": True,
        "free_disk_bytes": free,
        "minimum_free_disk_bytes": MIN_FREE_DISK_BYTES,
        "maximum_checkpoint_bytes_each": MAX_CHECKPOINT_BYTES,
        "simultaneous_checkpoint_files": 3,
        "expected_peak_checkpoint_bytes": EXPECTED_PEAK_CHECKPOINT_BYTES,
        "disk_headroom_after_expected_peak_bytes": free - EXPECTED_PEAK_CHECKPOINT_BYTES,
        "source_file_count": len(binding["files"]),
        "protocol_identity_sha256": binding["protocol_identity_sha256"],
        "source_bundle_identity_sha256": binding["source_bundle_identity_sha256"],
        "minimum_replay_cpu_wall_ratio": MIN_REPLAY_CPU_WALL_RATIO,
        "held_out_data_accessed": False,
        "execution_started": False,
    }


def _publication_record(publication: TemporalCheckpointPublication) -> dict[str, object]:
    return {
        "outcome": publication.outcome.name,
        "semantically_resumable": publication.semantically_resumable,
        "checkpoint_bytes": publication.checkpoint_bytes,
        "encoded_bytes": publication.encoded_bytes,
        "temp_cleanup": publication.temp_cleanup.name,
    }


def _block_distribution(values: Sequence[int]) -> dict[str, float | int]:
    return cn2._distribution(values)


def _run_tokens(
    circuit: NativeTemporalCircuit,
    tokens: np.ndarray,
    *,
    mode: TemporalStepMode,
    reset_before: bool,
    deadline_ns: int,
) -> dict[str, object]:
    block_latencies: list[int] = []
    per_token_ns: list[int] = []
    allocated = 0
    punished = 0
    predicted_columns = 0
    replay_started_ns = time.perf_counter_ns()
    replay_process_started_ns = time.process_time_ns()
    blocks = 0
    for start in range(0, int(tokens.size), BLOCK_TOKENS):
        block = tokens[start : start + BLOCK_TOKENS]
        normalized = tuple(int(token) for token in block)
        started_ns = time.perf_counter_ns()
        steps = circuit.advance_block(
            normalized,
            mode=mode,
            reset_before=reset_before and start == 0,
        )
        latency_ns = time.perf_counter_ns() - started_ns
        if len(steps) != len(normalized):
            raise RuntimeError("native dense block returned an incomplete step tuple")
        block_latencies.append(latency_ns)
        per_token_ns.append(math.ceil(latency_ns / len(normalized)))
        allocated += sum(step.allocated_segments for step in steps)
        punished += sum(step.punished_segments for step in steps)
        predicted_columns += sum(step.predicted_columns for step in steps)
        blocks += 1
        if time.perf_counter_ns() > deadline_ns:
            raise RuntimeError("dense cost run exceeded its six-hour wall cap")
    replay_wall_ns = time.perf_counter_ns() - replay_started_ns
    replay_process_ns = time.process_time_ns() - replay_process_started_ns
    return {
        "tokens": int(tokens.size),
        "blocks": blocks,
        "block_tokens": BLOCK_TOKENS,
        "replay_wall_ns": replay_wall_ns,
        "replay_process_ns": replay_process_ns,
        "replay_cpu_wall_ratio": replay_process_ns / replay_wall_ns,
        "steps_per_second": int(tokens.size) * 1_000_000_000.0 / replay_wall_ns,
        "block_latency_ns": _block_distribution(block_latencies),
        "normalized_ns_per_token_by_block": _block_distribution(per_token_ns),
        "actual_allocated_segments": allocated,
        "punished_segments": punished,
        "predicted_columns_total": predicted_columns,
    }


def _build_train_chunk(
    definition: object,
    books: Sequence[object],
    token_arrays: Sequence[np.ndarray],
    *,
    base_checkpoint: Path,
    manifest_path: Path,
    authority: dict[str, object],
    binding: dict[str, object],
    max_books: int,
    deadline_ns: int,
    invocation_setup_wall_ns: int = 0,
    invocation_setup_process_ns: int = 0,
) -> dict[str, object]:
    if max_books <= 0 or len(books) != len(token_arrays) or not books:
        raise ValueError("invalid bounded TRAIN chunk")

    started_ns = time.perf_counter_ns()
    process_started_ns = time.process_time_ns()
    phase_ns = time.perf_counter_ns()
    previous = _validate_build_manifest(manifest_path, authority, books)
    validation_ns = time.perf_counter_ns() - phase_ns
    if previous is not None and previous["status"] == "BUILD_COMPLETE":
        return previous

    start = 0 if previous is None else int(previous["cursor"]["next_book_ordinal"])
    end = min(start + max_books, len(books))
    old_checkpoint = (
        None if previous is None else _manifest_checkpoint_path(manifest_path, previous)
    )
    new_checkpoint = _partial_checkpoint_path(base_checkpoint, end)
    if new_checkpoint.exists():
        orphan_bytes = new_checkpoint.stat().st_size
        new_checkpoint.unlink()
        print(f"removed orphan {new_checkpoint.name} ({orphan_bytes:,} bytes)", flush=True)

    rows = [] if previous is None else list(previous["books"])
    allocated = 0 if previous is None else int(previous["actual_allocated_segments"])
    with _cost_authority(binding):
        phase_ns = time.perf_counter_ns()
        if old_checkpoint is None:
            circuit = NativeTemporalCircuit.fresh(definition)
            open_kind = "FRESH"
        else:
            circuit = NativeTemporalCircuit.from_checkpoint_path(definition, old_checkpoint)
            open_kind = "RESUME_IMPORT"
        open_ns = time.perf_counter_ns() - phase_ns
        try:
            if (
                previous is not None
                and asdict(circuit.checkpoint_identity()) != previous["checkpoint"]["identity"]
            ):
                raise RuntimeError("resumed checkpoint identity drifted")
            for ordinal in range(start, end):
                book, tokens = books[ordinal], token_arrays[ordinal]
                if int(tokens.size) != int(book.token_count):
                    raise RuntimeError("TRAIN book token materialization drifted")
                measurement = _run_tokens(
                    circuit,
                    tokens,
                    mode=TemporalStepMode.LEARN,
                    reset_before=True,
                    deadline_ns=deadline_ns,
                )
                allocated += int(measurement["actual_allocated_segments"])
                rows.append(
                    {
                        "ordinal": ordinal,
                        "book": book.short_name,
                        **measurement,
                        "cumulative_actual_allocated_segments": allocated,
                    }
                )
                print(
                    f"TRAIN49 {ordinal + 1:02d}/{len(books):02d} {book.short_name}: "
                    f"segments={allocated:,} wall={measurement['replay_wall_ns'] / 1e9:.2f}s",
                    flush=True,
                )
            terminal = circuit.checkpoint_identity()
            if terminal.segment_count != allocated:
                raise RuntimeError("terminal segments differ from actual allocations")
            phase_ns = time.perf_counter_ns()
            publication = circuit.publish_checkpoint(new_checkpoint)
            export_ns = time.perf_counter_ns() - phase_ns
            if not publication.semantically_resumable:
                raise RuntimeError("partial checkpoint publication was not durable")
        finally:
            circuit.close()

    checkpoint_sha256, hash_ns = _path_sha256(new_checkpoint)
    replay_ns = sum(int(row["replay_wall_ns"]) for row in rows[start:end])
    replay_process_ns = sum(int(row["replay_process_ns"]) for row in rows[start:end])
    checkpoint_io_ns = validation_ns + open_ns + export_ns + hash_ns
    build_ns = invocation_setup_wall_ns + time.perf_counter_ns() - started_ns
    build_process_ns = invocation_setup_process_ns + time.process_time_ns() - process_started_ns
    prior_timing = {} if previous is None else previous["timing"]

    def total(name: str, value: int) -> int:
        return int(prior_timing.get(name, 0)) + value

    chunk = {
        "books": [start, end],
        "open_kind": open_kind,
        "setup_wall_ns": invocation_setup_wall_ns,
        "native_open_wall_ns": open_ns,
        "replay_wall_ns": replay_ns,
        "checkpoint_io_wall_ns": checkpoint_io_ns,
        "checkpoint_export_wall_ns": export_ns,
        "build_wall_ns": build_ns,
        "cpu_wall_ratio": build_process_ns / build_ns,
    }
    manifest: dict[str, object] = {
        "schema_version": PARTIAL_MANIFEST_SCHEMA_VERSION,
        "artifact_id": ARTIFACT_ID,
        "status": "BUILD_COMPLETE" if end == len(books) else "BUILD_PARTIAL",
        "authority": authority,
        "cursor": {
            "next_book_ordinal": end,
            "token_count": sum(int(book.token_count) for book in books[:end]),
        },
        "actual_allocated_segments": allocated,
        "books": rows,
        "timing": {
            "cumulative_build_wall_ns": total("cumulative_build_wall_ns", build_ns),
            "cumulative_build_process_ns": total("cumulative_build_process_ns", build_process_ns),
            "cumulative_replay_wall_ns": total("cumulative_replay_wall_ns", replay_ns),
            "cumulative_replay_process_ns": total(
                "cumulative_replay_process_ns", replay_process_ns
            ),
            "cumulative_checkpoint_io_wall_ns": total(
                "cumulative_checkpoint_io_wall_ns", checkpoint_io_ns
            ),
            "chunks": [*prior_timing.get("chunks", []), chunk],
        },
        "checkpoint": {
            "file": new_checkpoint.name,
            "bytes": new_checkpoint.stat().st_size,
            "sha256": checkpoint_sha256,
            "identity": asdict(terminal),
            "publication": _publication_record(publication),
        },
    }
    _atomic_write_json(manifest_path, manifest)
    if old_checkpoint is not None:
        old_checkpoint.unlink()
    return manifest


def _load_train() -> tuple[object, tuple[object, ...], tuple[np.ndarray, ...]]:
    verified = load_verified_train_input()
    books = tuple(verified.iter_books())
    if (
        verified.token_count != cn2.FULL_TRAIN_TOKENS
        or verified.train_stream_sha256 != cn2.FULL_TRAIN_STREAM_SHA256
        or len(books) != cn2.FULL_TRAIN_BOOKS
        or books[0].short_name != "Genesis"
        or books[0].token_count != GENESIS_TOKENS
    ):
        raise RuntimeError("canonical public TRAIN authority drifted")
    arrays = tuple(np.frombuffer(book.token_bytes_le_i32, dtype="<i4") for book in books)
    if sum(int(tokens.size) for tokens in arrays) != cn2.FULL_TRAIN_TOKENS:
        raise RuntimeError("materialized TRAIN token count drifted")
    return verified, books, arrays


def _project_schedule(
    *,
    setup_ns: int,
    epoch0_replay_ns: int,
    dense_learn_wall_ns: int,
    dense_readonly_wall_ns: int,
    maximum_export_ns: int,
    maximum_import_ns: int,
) -> dict[str, object]:
    dense_ns_per_token = max(dense_learn_wall_ns, dense_readonly_wall_ns) / GENESIS_TOKENS
    projected_ns = (
        SERIALIZED_ARMS * setup_ns
        + SERIALIZED_ARMS * epoch0_replay_ns
        + SERIALIZED_ARMS * DENSE_STEPS_AFTER_EPOCH0_PER_ARM * dense_ns_per_token
        + SERIALIZED_ARMS * CHECKPOINT_EXPORTS_PER_ARM * maximum_export_ns
        + SERIALIZED_ARMS * RESUME_IMPORTS_PER_ARM * maximum_import_ns
    )
    projected_hours = projected_ns / 3_600_000_000_000
    return {
        "authorizes_c1": False,
        "classification": "direct full-density developmental projection; not launch evidence",
        "setup_ns_per_arm": setup_ns,
        "measured_epoch0_train49_replay_ns_per_arm": epoch0_replay_ns,
        "dense_genesis_conservative_ns_per_token": dense_ns_per_token,
        "dense_steps_after_epoch0_per_arm": DENSE_STEPS_AFTER_EPOCH0_PER_ARM,
        "checkpoint_exports_per_arm": CHECKPOINT_EXPORTS_PER_ARM,
        "resume_imports_per_arm": RESUME_IMPORTS_PER_ARM,
        "maximum_measured_export_ns": maximum_export_ns,
        "maximum_measured_import_ns": maximum_import_ns,
        "serialized_arms": SERIALIZED_ARMS,
        "schedule_steps": cn2.TOTAL_C1_SCHEDULE_STEPS,
        "projected_total_hours": projected_hours,
        "development_cap_hours": 13.5,
        "passes_development_cap": projected_hours < 13.5,
        "accounting_rule": (
            "three measured epoch-0 TRAIN49 replays plus dense terminal-state rate "
            "for the remaining 8,927,839 steps/arm, five exports/arm, and one "
            "resume-import allowance/arm"
        ),
        "model_limit": (
            "one arm's TRAIN49 build and Genesis full-density clones; no protected "
            "execution, capability, or launch evidence"
        ),
    }


def _timing_verdict(
    projection: dict[str, object], replay_cpu_wall_ratios: Sequence[float]
) -> tuple[str, dict[str, object]]:
    minimum = min(replay_cpu_wall_ratios)
    quality = {
        "clock": "time.process_time_ns / time.perf_counter_ns",
        "minimum_required_ratio": MIN_REPLAY_CPU_WALL_RATIO,
        "observed_ratios": list(replay_cpu_wall_ratios),
        "minimum_observed_ratio": minimum,
        "contention_stop": minimum < MIN_REPLAY_CPU_WALL_RATIO,
        "cap_verdict_authoritative": minimum >= MIN_REPLAY_CPU_WALL_RATIO,
    }
    if quality["contention_stop"]:
        return "STOP_P2_COST_DENSE_TRAIN49_CONTENTION", quality
    if projection["passes_development_cap"]:
        return "PASS_P2_COST_DENSE_TRAIN49_DEVELOPMENT", quality
    return "STOP_P2_COST_DENSE_TRAIN49_CAP", quality


def _measure_clone(
    definition: object,
    base_checkpoint: Path,
    clone_checkpoint: Path,
    genesis: np.ndarray,
    *,
    mode: TemporalStepMode,
    base_identity: object,
    deadline_ns: int,
) -> dict[str, object]:
    import_started_ns = time.perf_counter_ns()
    circuit = NativeTemporalCircuit.from_checkpoint_path(definition, base_checkpoint)
    import_ns = time.perf_counter_ns() - import_started_ns
    try:
        reopened = circuit.checkpoint_identity()
        if reopened != base_identity:
            raise RuntimeError("dense clone did not reopen the exact base checkpoint")
        replay = _run_tokens(
            circuit,
            genesis,
            mode=mode,
            reset_before=True,
            deadline_ns=deadline_ns,
        )
        terminal = circuit.checkpoint_identity()
        if mode is TemporalStepMode.READ_ONLY and (
            replay["actual_allocated_segments"] != 0
            or replay["punished_segments"] != 0
            or terminal.segment_count != base_identity.segment_count
            or terminal.learning_step != base_identity.learning_step
        ):
            raise RuntimeError("READ_ONLY dense clone mutated learned-state accounting")
        export_started_ns = time.perf_counter_ns()
        publication = circuit.publish_checkpoint(clone_checkpoint)
        export_ns = time.perf_counter_ns() - export_started_ns
        if not publication.semantically_resumable:
            raise RuntimeError("dense clone checkpoint publication was not durable")
    finally:
        circuit.close()
    checkpoint_sha256, hash_ns = _path_sha256(clone_checkpoint)
    return {
        "mode": mode.name,
        "import_ns": import_ns,
        "replay": replay,
        "terminal_checkpoint_identity": asdict(terminal),
        "export_ns": export_ns,
        "publication": _publication_record(publication),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_hash_ns": hash_ns,
    }


def _measure_complete_build(
    output: Path,
    base_checkpoint: Path,
    interpretation: dict[str, object],
) -> dict[str, object]:
    del output
    active_started_ns = time.perf_counter_ns()
    deadline_ns = active_started_ns + RUN_WALL_CAP_NS
    verified, books, token_arrays = _load_train()
    protocol, definition = cn2._definition()
    binding = _authority_binding()
    authority = _build_authority(definition, binding)
    manifest_path = _partial_manifest_path(base_checkpoint)
    build_manifest = _validate_build_manifest(manifest_path, authority, books)
    if build_manifest is None or build_manifest["status"] != "BUILD_COMPLETE":
        raise RuntimeError("dense measurement requires an authenticated complete TRAIN49 build")
    active_checkpoint = _manifest_checkpoint_path(manifest_path, build_manifest)
    build_timing = build_manifest["timing"]
    chunk_rows = build_timing["chunks"]
    if not chunk_rows or chunk_rows[0]["open_kind"] != "FRESH":
        raise RuntimeError("complete TRAIN49 build lacks its fresh-construction timing")

    with _cost_authority(binding):
        base_import_started_ns = time.perf_counter_ns()
        base_import_process_started_ns = time.process_time_ns()
        with NativeTemporalCircuit.from_checkpoint_path(definition, active_checkpoint) as circuit:
            base_import_ns = time.perf_counter_ns() - base_import_started_ns
            base_import_process_ns = time.process_time_ns() - base_import_process_started_ns
            base_identity = circuit.checkpoint_identity()
        if asdict(base_identity) != build_manifest["checkpoint"]["identity"]:
            raise RuntimeError("complete TRAIN49 checkpoint identity drifted at measurement")

        with tempfile.TemporaryDirectory(
            prefix="p2-cost-dense-train49-", dir=base_checkpoint.parent
        ) as raw_scratch:
            scratch = Path(raw_scratch)
            learn_clone = _measure_clone(
                definition,
                active_checkpoint,
                scratch / "genesis-learn.checkpoint",
                token_arrays[0],
                mode=TemporalStepMode.LEARN,
                base_identity=base_identity,
                deadline_ns=deadline_ns,
            )
            readonly_clone = _measure_clone(
                definition,
                active_checkpoint,
                scratch / "genesis-readonly.checkpoint",
                token_arrays[0],
                mode=TemporalStepMode.READ_ONLY,
                base_identity=base_identity,
                deadline_ns=deadline_ns,
            )
        native_identity = loaded_library_identity()

    active_wall_ns = time.perf_counter_ns() - active_started_ns
    actual_allocations = int(build_manifest["actual_allocated_segments"])
    epoch0_replay_ns = int(build_timing["cumulative_replay_wall_ns"])
    epoch0_process_ns = int(build_timing["cumulative_replay_process_ns"])
    chunk_export_ns = [int(chunk["checkpoint_export_wall_ns"]) for chunk in chunk_rows]
    chunk_resume_import_ns = [
        int(chunk["native_open_wall_ns"])
        for chunk in chunk_rows
        if chunk["open_kind"] == "RESUME_IMPORT"
    ]
    maximum_export_ns = max(
        *chunk_export_ns,
        int(learn_clone["export_ns"]),
        int(readonly_clone["export_ns"]),
    )
    maximum_import_ns = max(
        base_import_ns,
        *chunk_resume_import_ns,
        int(learn_clone["import_ns"]),
        int(readonly_clone["import_ns"]),
    )
    projection = _project_schedule(
        setup_ns=(int(chunk_rows[0]["setup_wall_ns"]) + int(chunk_rows[0]["native_open_wall_ns"])),
        epoch0_replay_ns=epoch0_replay_ns,
        dense_learn_wall_ns=int(learn_clone["replay"]["replay_wall_ns"]),
        dense_readonly_wall_ns=int(readonly_clone["replay"]["replay_wall_ns"]),
        maximum_export_ns=maximum_export_ns,
        maximum_import_ns=maximum_import_ns,
    )
    status, timing_quality = _timing_verdict(
        projection,
        (
            epoch0_process_ns / epoch0_replay_ns,
            float(learn_clone["replay"]["replay_cpu_wall_ratio"]),
            float(readonly_clone["replay"]["replay_cpu_wall_ratio"]),
        ),
    )
    return {
        "schema_version": 1,
        "artifact_id": ARTIFACT_ID,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": status,
        "claim_boundary": GATE_CONTRACT["claim_boundary"],
        "isolation": {
            "orchestrator_confirmed_no_other_heavy_process": True,
            "measurement_process_id": os.getpid(),
        },
        "input": {
            "partition": "public TRAIN",
            "candidate_id": verified.candidate_id,
            "train_stream_sha256": verified.train_stream_sha256,
            "train_books_materialized": len(books),
            "train_tokens": verified.token_count,
            "held_out_books_materialized": 0,
        },
        "definition": {
            "arm": list(cn2.ARM),
            "construction_seed": cn2.CONSTRUCTION_SEED,
            "definition_protocol_sha256": cn2.definition_protocol.protocol_sha256(),
            "definition_protocol_runnable": protocol["source_binding"]["runnable"],
            "segment_capacity": definition.config.segment_capacity,
            "threshold_selected_by_this_gate": False,
        },
        "train49_setup": {
            "identity_rescored": False,
            "accounting_rule": "terminal segments equal sum of native actual allocations",
            "actual_allocated_segments": actual_allocations,
            "terminal_segment_count": base_identity.segment_count,
            "setup_pre_native_ns": int(chunk_rows[0]["setup_wall_ns"]),
            "native_construction_ns": int(chunk_rows[0]["native_open_wall_ns"]),
            "epoch0_replay_ns": epoch0_replay_ns,
            "epoch0_replay_process_ns": epoch0_process_ns,
            "epoch0_replay_cpu_wall_ratio": epoch0_process_ns / epoch0_replay_ns,
            "books": build_manifest["books"],
            "engineering_build_timing": build_timing,
            "partial_manifest_path": manifest_path.relative_to(REPO_ROOT).as_posix(),
            "base_checkpoint_identity": asdict(base_identity),
            "base_checkpoint_path": active_checkpoint.relative_to(REPO_ROOT).as_posix(),
            "base_checkpoint_sha256": build_manifest["checkpoint"]["sha256"],
            "base_export_ns": chunk_rows[-1]["checkpoint_export_wall_ns"],
            "base_publication": build_manifest["checkpoint"]["publication"],
            "measurement_reopen_import_ns": base_import_ns,
            "measurement_reopen_import_process_ns": base_import_process_ns,
        },
        "dense_genesis": {
            "token_count": GENESIS_TOKENS,
            "learn_clone": learn_clone,
            "read_only_clone": readonly_clone,
        },
        "developmental_projection": projection,
        "timing_quality": timing_quality,
        "source_binding": binding,
        "lineage": {
            "cn2_interpretation_path": CN2_INTERPRETATION_PATH.relative_to(REPO_ROOT).as_posix(),
            "cn2_interpretation_classification": interpretation["classification"],
        },
        "active_wall_ns": active_wall_ns,
        "engineering_build_plus_measurement_wall_ns": (
            int(build_timing["cumulative_build_wall_ns"]) + active_wall_ns
        ),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "native": native_identity.to_record(),
        },
    }


def main() -> int:
    args = _parse_args()
    output = args.output.resolve()
    base_checkpoint = args.base_checkpoint.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    base_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    interpretation, preflight = _preflight(output, base_checkpoint)
    if args.preflight_only:
        print(json.dumps(preflight, indent=2, sort_keys=True), flush=True)
        return 0
    if args.build_chunk_books is not None:
        setup_started_ns = time.perf_counter_ns()
        setup_process_started_ns = time.process_time_ns()
        _verified, books, token_arrays = _load_train()
        _protocol, definition = cn2._definition()
        binding = _authority_binding()
        authority = _build_authority(definition, binding)
        setup_wall_ns = time.perf_counter_ns() - setup_started_ns
        setup_process_ns = time.process_time_ns() - setup_process_started_ns
        manifest_path = _partial_manifest_path(base_checkpoint)
        manifest = _build_train_chunk(
            definition,
            books,
            token_arrays,
            base_checkpoint=base_checkpoint,
            manifest_path=manifest_path,
            authority=authority,
            binding=binding,
            max_books=args.build_chunk_books,
            deadline_ns=time.perf_counter_ns() + RUN_WALL_CAP_NS,
            invocation_setup_wall_ns=setup_wall_ns,
            invocation_setup_process_ns=setup_process_ns,
        )
        print(
            f"{manifest['status']}: books={manifest['cursor']['next_book_ordinal']}/"
            f"{len(books)} manifest={manifest_path}",
            flush=True,
        )
        return 0
    payload = _measure_complete_build(output, base_checkpoint, interpretation)
    _atomic_write_json(output, payload)
    print(f"{payload['status']}: {output}", flush=True)
    return 0 if str(payload["status"]).startswith("PASS_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
