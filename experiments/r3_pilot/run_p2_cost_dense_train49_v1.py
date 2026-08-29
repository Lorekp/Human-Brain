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
        "one exact 49-book TRAIN LEARN pass; record only native actual allocations "
        "and require terminal segment_count to equal their sum"
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
    args = parser.parse_args()
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
    if output.exists() or checkpoint.exists():
        raise RuntimeError("dense cost output already exists; refusing overwrite")
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
    return interpretation, {
        "status": "PREFLIGHT_READY_NOT_RUN",
        "artifact_id": ARTIFACT_ID,
        "output": str(output),
        "base_checkpoint": str(checkpoint),
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


def _run(
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
    setup_pre_native_ns = time.perf_counter_ns() - active_started_ns

    book_rows: list[dict[str, object]] = []
    actual_allocations = 0
    with _cost_authority(binding):
        construction_started_ns = time.perf_counter_ns()
        with NativeTemporalCircuit.fresh(definition) as circuit:
            construction_ns = time.perf_counter_ns() - construction_started_ns
            epoch0_started_ns = time.perf_counter_ns()
            epoch0_process_started_ns = time.process_time_ns()
            for ordinal, (book, tokens) in enumerate(zip(books, token_arrays, strict=True)):
                measurement = _run_tokens(
                    circuit,
                    tokens,
                    mode=TemporalStepMode.LEARN,
                    reset_before=True,
                    deadline_ns=deadline_ns,
                )
                actual_allocations += int(measurement["actual_allocated_segments"])
                row = {
                    "ordinal": ordinal,
                    "book": book.short_name,
                    **measurement,
                    "cumulative_actual_allocated_segments": actual_allocations,
                }
                book_rows.append(row)
                print(
                    f"TRAIN49 {ordinal + 1:02d}/49 {book.short_name}: "
                    f"segments={actual_allocations:,} "
                    f"wall={measurement['replay_wall_ns'] / 1e9:.2f}s",
                    flush=True,
                )
            epoch0_replay_ns = time.perf_counter_ns() - epoch0_started_ns
            epoch0_process_ns = time.process_time_ns() - epoch0_process_started_ns
            base_identity = circuit.checkpoint_identity()
            if base_identity.segment_count != actual_allocations:
                raise RuntimeError(
                    "full TRAIN terminal segment count differs from actual allocations"
                )
            export_started_ns = time.perf_counter_ns()
            base_publication = circuit.publish_checkpoint(base_checkpoint)
            base_export_ns = time.perf_counter_ns() - export_started_ns
            if not base_publication.semantically_resumable:
                raise RuntimeError("full TRAIN checkpoint publication was not durable")
        base_sha256, base_hash_ns = _path_sha256(base_checkpoint)

        with tempfile.TemporaryDirectory(
            prefix="p2-cost-dense-train49-", dir=base_checkpoint.parent
        ) as raw_scratch:
            scratch = Path(raw_scratch)
            learn_clone = _measure_clone(
                definition,
                base_checkpoint,
                scratch / "genesis-learn.checkpoint",
                token_arrays[0],
                mode=TemporalStepMode.LEARN,
                base_identity=base_identity,
                deadline_ns=deadline_ns,
            )
            readonly_clone = _measure_clone(
                definition,
                base_checkpoint,
                scratch / "genesis-readonly.checkpoint",
                token_arrays[0],
                mode=TemporalStepMode.READ_ONLY,
                base_identity=base_identity,
                deadline_ns=deadline_ns,
            )
        native_identity = loaded_library_identity()

    active_wall_ns = time.perf_counter_ns() - active_started_ns
    maximum_export_ns = max(
        base_export_ns,
        int(learn_clone["export_ns"]),
        int(readonly_clone["export_ns"]),
    )
    maximum_import_ns = max(int(learn_clone["import_ns"]), int(readonly_clone["import_ns"]))
    projection = _project_schedule(
        setup_ns=setup_pre_native_ns + construction_ns,
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
            "setup_pre_native_ns": setup_pre_native_ns,
            "native_construction_ns": construction_ns,
            "epoch0_replay_ns": epoch0_replay_ns,
            "epoch0_replay_process_ns": epoch0_process_ns,
            "epoch0_replay_cpu_wall_ratio": epoch0_process_ns / epoch0_replay_ns,
            "books": book_rows,
            "base_checkpoint_identity": asdict(base_identity),
            "base_checkpoint_path": base_checkpoint.relative_to(REPO_ROOT).as_posix(),
            "base_checkpoint_sha256": base_sha256,
            "base_checkpoint_hash_ns": base_hash_ns,
            "base_export_ns": base_export_ns,
            "base_publication": _publication_record(base_publication),
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
    payload = _run(output, base_checkpoint, interpretation)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(f"{payload['status']}: {output}", flush=True)
    return 0 if str(payload["status"]).startswith("PASS_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
