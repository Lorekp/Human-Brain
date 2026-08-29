"""Resumable public-TRAIN Path-2 factorized-successor replay.

This separately named adjudication starts from empty native and scorer state.
It observes only the authenticated 49-book public TRAIN stream, scores complete
``(source winners, token)`` tuples and exact-source per-column owner reuse, and
records matching-versus-active counters as non-gating telemetry.  Book-boundary
native checkpoints are paired with compact scorer journals and an atomic cursor
manifest.  No completed dense checkpoint, held-out book, or C1 input is accepted.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import platform
import shutil
import struct
import sys
import tempfile
import time
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, BinaryIO

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import humanbrain.native_temporal_runtime as runtime_adapter  # noqa: E402, I001
from experiments.r3_pilot import factorized_successor_scoring as scoring  # noqa: E402
from experiments.r3_pilot import run_cn2_dev_4096 as cn2  # noqa: E402
from experiments.r3_pilot.r3_f0_train_input import (  # noqa: E402
    load_verified_train_input,
)
from humanbrain.native import loaded_library_identity  # noqa: E402
from humanbrain.native_temporal_runtime import (  # noqa: E402
    NativeTemporalCircuit,
    NativeTemporalCircuitDefinition,
    TemporalCheckpointPublication,
    TemporalStepMode,
)


ARTIFACT_ID = "P2-FACTORIZED-SUCCESSOR-TRAIN49-REPLAY-v1"
OUTPUT_PATH = Path(__file__).with_name(
    "p2_factorized_successor_train49_replay_v1.json"
)
STATE_BASE_PATH = Path(__file__).with_name(
    "p2_factorized_successor_train49_replay_v1"
)
ARM = cn2.ARM
CONSTRUCTION_SEED = cn2.CONSTRUCTION_SEED
ACTIVE_COLUMNS = scoring.ACTIVE_COLUMNS
BLOCK_TOKENS = 256
MAX_CHUNK_BOOKS = 2
MANIFEST_SCHEMA_VERSION = 1
SPEND_SCHEMA_VERSION = 1
SHARD_SCHEMA_VERSION = 1
RUN_WALL_CAP_NS = 3 * 60 * 60 * 1_000_000_000

# Durable scorer state contains only fields retained by FactorizedSuccessorScorer.
# Token columns, event counts, reset positions, candidate counts, and postings are
# reconstructed from the source-bound codebook and authenticated book boundaries.
SCORER_RECORD_FIELDS = (
    "token_id",
    "matching_segments",
    "active_segments",
    "punished_segments",
    "allocated_segments",
    "predicted_columns",
    "winner_cells[11]",
)
SCORER_RECORD = struct.Struct("<4I2B11I")
SCORER_SHARD_MAGIC = b"HBFSJ1\0\0"
SCORER_SHARD_HEADER = struct.Struct("<8sIIQ32s")
SCORER_RECORD_BYTES = SCORER_RECORD.size
SCORER_SHARD_HEADER_BYTES = SCORER_SHARD_HEADER.size

MAX_NATIVE_CHECKPOINT_BYTES = (
    256
    + 64 * cn2.WORST_CASE_SEGMENT_CAPACITY
    + 4 * 12_288
    + 2 * 22 * 4
)
MAX_SCORER_JOURNAL_BYTES = (
    cn2.FULL_TRAIN_TOKENS * SCORER_RECORD_BYTES
    + cn2.FULL_TRAIN_BOOKS * SCORER_SHARD_HEADER_BYTES
)
MAX_BOOK_TOKENS = 122_253
EXPECTED_FINAL_DURABLE_BYTES = (
    MAX_NATIVE_CHECKPOINT_BYTES + MAX_SCORER_JOURNAL_BYTES
)
EXPECTED_PEAK_DURABLE_BYTES = (
    2 * MAX_NATIVE_CHECKPOINT_BYTES
    + MAX_SCORER_JOURNAL_BYTES
    + SCORER_SHARD_HEADER_BYTES
    + MAX_BOOK_TOKENS * SCORER_RECORD_BYTES
)
MIN_FREE_DISK_BYTES = 4 * 1024**3
REFERENCE_DENSE_NATIVE_TRAIN49_REPLAY_NS = 5_656_864_671_400
EXPECTED_REPLAY_HOURS = {"lower": 2.0, "upper": 4.0}

CLAIM_BOUNDARY = (
    "one source-bound resumable replay from empty native/scorer state over the "
    "canonical public 49-book TRAIN stream at arm (16,18), seed 17; complete-"
    "tuple and per-column successor identity gate the verdict while matching "
    "and active segments remain separate non-gating telemetry; no completed-"
    "build reuse, calibration retry, threshold choice, held-out/C1 access, "
    "capability, transfer, or launch evidence"
)

GATE_CONTRACT: dict[str, object] = {
    "artifact_id": ARTIFACT_ID,
    "partition": "public TRAIN only",
    "books": cn2.FULL_TRAIN_BOOKS,
    "tokens": cn2.FULL_TRAIN_TOKENS,
    "train_stream_sha256": cn2.FULL_TRAIN_STREAM_SHA256,
    "arm": list(ARM),
    "construction_seed": CONSTRUCTION_SEED,
    "native_start": "fresh empty state only",
    "scorer_start": "fresh empty state only",
    "identity_criterion": (
        "stable complete (source winner tuple, observed token) keys plus stable "
        "owners for every exact-source output-column factor"
    ),
    "allocation_criterion": (
        "sum native actual allocations; never require 11 times novel keys"
    ),
    "telemetry": "matching and active segment counters are separate and non-gating",
    "resume": (
        "one active authenticated native checkpoint plus per-book compact scorer "
        "journals under an atomic cursor manifest"
    ),
    "claim_boundary": {
        "authorizes_c1": False,
        "capability_evidence": False,
        "completed_build_reused": False,
        "held_out_books_materialized": 0,
        "threshold_selection": False,
        "transfer_evidence": False,
    },
}


@dataclass(frozen=True, slots=True)
class ReplayPaths:
    """All output paths derived from one explicit sibling-only state base."""

    output: Path
    state_base: Path

    @property
    def marker(self) -> Path:
        return self.state_base.with_name(f"{self.state_base.name}.spend.json")

    @property
    def manifest(self) -> Path:
        return self.state_base.with_name(f"{self.state_base.name}.partial.json")

    def native_checkpoint(self, completed_books: int) -> Path:
        if not 0 <= completed_books <= cn2.FULL_TRAIN_BOOKS:
            raise ValueError("completed_books")
        return self.state_base.with_name(
            f"{self.state_base.name}.native.book-{completed_books:02d}.checkpoint"
        )

    def scorer_shard(self, book_ordinal: int) -> Path:
        if not 0 <= book_ordinal < cn2.FULL_TRAIN_BOOKS:
            raise ValueError("book_ordinal")
        return self.state_base.with_name(
            f"{self.state_base.name}.scorer.book-{book_ordinal + 1:02d}.bin"
        )


DEFAULT_PATHS = ReplayPaths(OUTPUT_PATH, STATE_BASE_PATH)


class ReplayAdjudicationStop(RuntimeError):
    """The first chronological scorer contract failure in a returned block."""

    def __init__(self, record: dict[str, object]) -> None:
        super().__init__(str(record["scorer_stop_kind"]))
        self.record = record


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _path_sha256(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest(), path.stat().st_size


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temp = tempfile.mkstemp(
        prefix=f"{path.name}.", suffix=".tmp", dir=path.parent
    )
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


def _exclusive_write_json(path: Path, payload: dict[str, object]) -> None:
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    try:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as error:
        raise RuntimeError(f"refusing to overwrite {path.name}") from error


def _declared_state_residue(paths: ReplayPaths) -> tuple[Path, ...]:
    """Return exact native/scorer siblings that have no manifest authority."""

    declared = (
        *(paths.native_checkpoint(cursor) for cursor in range(cn2.FULL_TRAIN_BOOKS + 1)),
        *(paths.scorer_shard(ordinal) for ordinal in range(cn2.FULL_TRAIN_BOOKS)),
    )
    exact = [path for path in declared if path.exists()]
    temporary: list[Path] = []
    for pattern in (
        f"{paths.state_base.name}.native.book-*.checkpoint.*.tmp",
        f"{paths.state_base.name}.scorer.book-*.bin.*.tmp",
    ):
        temporary.extend(paths.state_base.parent.glob(pattern))
    return tuple(sorted({*exact, *temporary}))


def _require_no_unmanifested_state(paths: ReplayPaths) -> None:
    residue = _declared_state_residue(paths)
    if residue:
        names = ", ".join(path.name for path in residue[:4])
        raise RuntimeError(
            "unmanifested native/scorer state residue blocks fresh spend: " + names
        )


def _source_binding() -> dict[str, object]:
    inherited = cn2._source_binding()
    files = dict(inherited["files"])
    for path in (Path(__file__).resolve(), Path(scoring.__file__).resolve()):
        files[path.relative_to(REPO_ROOT).as_posix()] = cn2._canonical_lf_sha256(path)
    source_sha256 = hashlib.sha256(
        _canonical_json_bytes(
            {
                "domain": "humanbrain-p2-factorized-successor-train49-replay-v1",
                "files": files,
            }
        )
    ).hexdigest()
    protocol_sha256 = hashlib.sha256(_canonical_json_bytes(GATE_CONTRACT)).hexdigest()
    return {
        "canonical_byte_rule": inherited["canonical_byte_rule"],
        "files": files,
        "protocol_identity_sha256": protocol_sha256,
        "source_bundle_identity_sha256": source_sha256,
    }


@contextlib.contextmanager
def _replay_authority(binding: dict[str, object]) -> Iterator[None]:
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


def _definition() -> tuple[dict[str, Any], NativeTemporalCircuitDefinition]:
    return cn2._definition()


def _load_train() -> tuple[object, tuple[object, ...], tuple[np.ndarray, ...]]:
    """Materialize only the VerifiedTrainInput book chunks."""

    verified = load_verified_train_input()
    books = tuple(verified.iter_books())
    if (
        verified.token_count != cn2.FULL_TRAIN_TOKENS
        or verified.train_stream_sha256 != cn2.FULL_TRAIN_STREAM_SHA256
        or len(books) != cn2.FULL_TRAIN_BOOKS
        or tuple(int(book.ordinal) for book in books)
        != tuple(range(cn2.FULL_TRAIN_BOOKS))
    ):
        raise RuntimeError("canonical public TRAIN authority drifted")
    arrays = tuple(np.frombuffer(book.token_bytes_le_i32, dtype="<i4") for book in books)
    if any(int(tokens.size) != int(book.token_count) for book, tokens in zip(books, arrays)):
        raise RuntimeError("public TRAIN book token materialization drifted")
    if sum(int(tokens.size) for tokens in arrays) != cn2.FULL_TRAIN_TOKENS:
        raise RuntimeError("public TRAIN token total drifted")
    return verified, books, arrays


def _native_identity_record(identity: object) -> dict[str, object]:
    return asdict(identity)


def _publication_record(publication: TemporalCheckpointPublication) -> dict[str, object]:
    return {
        "outcome": publication.outcome.name,
        "semantically_resumable": publication.semantically_resumable,
        "checkpoint_bytes": publication.checkpoint_bytes,
        "encoded_bytes": publication.encoded_bytes,
        "temp_cleanup": publication.temp_cleanup.name,
    }


def _authority_record(
    verified: object,
    definition: NativeTemporalCircuitDefinition,
    binding: dict[str, object],
) -> dict[str, object]:
    native = loaded_library_identity()
    return {
        "artifact_id": ARTIFACT_ID,
        "partition": "public TRAIN",
        "train_input_authority_sha256": verified.train_input_authority_sha256,
        "train_stream_sha256": verified.train_stream_sha256,
        "tokenizer_sha256": verified.tokenizer_sha256,
        "books": cn2.FULL_TRAIN_BOOKS,
        "tokens": cn2.FULL_TRAIN_TOKENS,
        "arm": list(ARM),
        "construction_seed": CONSTRUCTION_SEED,
        "definition_identity_sha256": definition.circuit_identity_sha256(),
        "protocol_identity_sha256": binding["protocol_identity_sha256"],
        "source_bundle_identity_sha256": binding["source_bundle_identity_sha256"],
        "native_source_digest": native.source_digest,
        "native_binary_sha256": native.binary_sha256,
        "mode": TemporalStepMode.LEARN.name,
        "reset_before_each_book": True,
        "start_state": "EMPTY_NATIVE_AND_SCORER",
        "scorer_record_format": SCORER_RECORD.format,
        "scorer_record_bytes": SCORER_RECORD_BYTES,
    }


def _authority_digest(authority: dict[str, object]) -> str:
    return hashlib.sha256(_canonical_json_bytes(authority)).hexdigest()


def _spend_payload(paths: ReplayPaths, authority: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": SPEND_SCHEMA_VERSION,
        "artifact_id": ARTIFACT_ID,
        "state": "FACTORIZED_SUCCESSOR_TRAIN49_REPLAY_SPENT",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "authority": authority,
        "authority_sha256": _authority_digest(authority),
        "criterion": GATE_CONTRACT,
        "claim_boundary": CLAIM_BOUNDARY,
        "output_file": paths.output.name,
        "manifest_file": paths.manifest.name,
        "exclusive_create": True,
        "flushed_and_fsynced_before_native_replay": True,
    }


def _create_spend_marker(paths: ReplayPaths, authority: dict[str, object]) -> dict[str, object]:
    if paths.output.exists() or paths.manifest.exists():
        raise RuntimeError("refusing fresh spend with an existing artifact or manifest")
    _require_no_unmanifested_state(paths)
    payload = _spend_payload(paths, authority)
    _exclusive_write_json(paths.marker, payload)
    return payload


def _validate_spend_marker(
    paths: ReplayPaths, authority: dict[str, object]
) -> dict[str, object]:
    try:
        marker = json.loads(paths.marker.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("spend marker is missing or unreadable") from error
    if (
        not isinstance(marker, dict)
        or marker.get("schema_version") != SPEND_SCHEMA_VERSION
        or marker.get("artifact_id") != ARTIFACT_ID
        or marker.get("state") != "FACTORIZED_SUCCESSOR_TRAIN49_REPLAY_SPENT"
        or marker.get("authority") != authority
        or marker.get("authority_sha256") != _authority_digest(authority)
        or marker.get("criterion") != GATE_CONTRACT
        or marker.get("claim_boundary") != CLAIM_BOUNDARY
        or marker.get("output_file") != paths.output.name
        or marker.get("manifest_file") != paths.manifest.name
        or marker.get("exclusive_create") is not True
        or marker.get("flushed_and_fsynced_before_native_replay") is not True
    ):
        raise RuntimeError("spend marker authority or contract drifted")
    return marker


def _load_or_create_spend_marker(
    paths: ReplayPaths, authority: dict[str, object]
) -> dict[str, object]:
    if paths.output.exists():
        raise RuntimeError("factorized successor replay is already terminal")
    if not paths.marker.exists():
        return _create_spend_marker(paths, authority)
    return _validate_spend_marker(paths, authority)


def _scorer_journal_contract() -> dict[str, object]:
    return {
        "schema_version": SHARD_SCHEMA_VERSION,
        "header_format": SCORER_SHARD_HEADER.format,
        "header_bytes_per_book": SCORER_SHARD_HEADER_BYTES,
        "record_format": SCORER_RECORD.format,
        "record_fields": list(SCORER_RECORD_FIELDS),
        "bytes_per_token_record": SCORER_RECORD_BYTES,
        "maximum_train_records": cn2.FULL_TRAIN_TOKENS,
        "maximum_total_bytes": MAX_SCORER_JOURNAL_BYTES,
        "derived_on_restore": {
            "columns": "source-bound immutable definition token code",
            "reset_before": "first record of each authenticated book shard",
            "event_count": "one-based record ordinal inside book shard",
            "candidate_segments": "matching_segments (not retained by scorer)",
            "postings_visited": "matching_segments (not retained by scorer)",
        },
        "durable_python_source_column_map": False,
    }


def _scorer_anchor(scorer: scoring.FactorizedSuccessorScorer) -> dict[str, object]:
    summary = scorer.summary()
    return {
        "observed_steps": summary["observed_steps"],
        "complete_key_count": summary["complete_key_count"],
        "actual_allocated_segments": summary["actual_allocated_segments"],
        "winner_trace_sha256_le_u32": summary["winner_trace_sha256_le_u32"],
    }


def _checkpoint_record(path: Path, identity: object, publication: object) -> dict[str, object]:
    digest, size = _path_sha256(path)
    return {
        "file": path.name,
        "bytes": size,
        "sha256": digest,
        "identity": _native_identity_record(identity),
        "publication": _publication_record(publication),
    }


def _manifest_status(cursor: int, book_count: int) -> str:
    if cursor == 0:
        return "REPLAY_EMPTY"
    if cursor == book_count:
        return "REPLAY_COMPLETE"
    return "REPLAY_PARTIAL"


def _manifest_sibling(path: Path, raw_name: object, label: str) -> Path:
    if not isinstance(raw_name, str):
        raise RuntimeError(f"{label} filename is invalid")
    lexical = Path(raw_name)
    if lexical.is_absolute() or lexical.name != raw_name:
        raise RuntimeError(f"{label} must be a sibling filename")
    return path.parent / lexical


def _initialize_empty_manifest(
    paths: ReplayPaths,
    definition: NativeTemporalCircuitDefinition,
    books: Sequence[object],
    *,
    authority: dict[str, object],
    binding: dict[str, object],
    marker: dict[str, object],
) -> dict[str, object]:
    if paths.manifest.exists():
        raise RuntimeError("empty replay manifest already exists")
    _require_no_unmanifested_state(paths)
    checkpoint_path = paths.native_checkpoint(0)
    scorer = scoring.FactorizedSuccessorScorer()
    with _replay_authority(binding):
        with NativeTemporalCircuit.fresh(definition) as circuit:
            identity = circuit.checkpoint_identity()
            if identity.segment_count != 0:
                raise RuntimeError("fresh replay native state is not empty")
            publication = circuit.publish_checkpoint(checkpoint_path)
    if not publication.semantically_resumable:
        raise RuntimeError("empty native checkpoint is not resumable")
    marker_sha256, _marker_bytes = _path_sha256(paths.marker)
    manifest: dict[str, object] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "artifact_id": ARTIFACT_ID,
        "status": "REPLAY_EMPTY",
        "authority": authority,
        "authority_sha256": _authority_digest(authority),
        "spend_marker": {
            "file": paths.marker.name,
            "sha256": marker_sha256,
            "state": marker["state"],
        },
        "start_state": {
            "native": "FRESH_EMPTY",
            "scorer": "FRESH_EMPTY",
            "completed_dense_checkpoint_imported": False,
        },
        "cursor": {"next_book_ordinal": 0, "token_count": 0},
        "books": [],
        "scorer_summary": scorer.summary(),
        "scorer_journal": _scorer_journal_contract(),
        "actual_allocated_segments": 0,
        "checkpoint": _checkpoint_record(checkpoint_path, identity, publication),
        "timing": {
            "cumulative_replay_wall_ns": 0,
            "cumulative_scorer_restore_wall_ns": 0,
            "cumulative_checkpoint_io_wall_ns": 0,
            "books": [],
        },
        "input_boundary": {
            "partition": "public TRAIN",
            "train_books_materialized": len(books),
            "held_out_books_materialized": 0,
            "c1_inputs_materialized": 0,
        },
    }
    _atomic_write_json(paths.manifest, manifest)
    return manifest


def _read_manifest(paths: ReplayPaths) -> dict[str, object]:
    try:
        value = json.loads(paths.manifest.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("replay manifest is missing or unreadable") from error
    if not isinstance(value, dict):
        raise RuntimeError("replay manifest must be a JSON object")
    return value


def _validate_hashed_file(path: Path, record: object, label: str) -> None:
    if not isinstance(record, dict):
        raise RuntimeError(f"{label} record is invalid")
    if not path.is_file():
        raise RuntimeError(f"{label} is missing")
    digest, size = _path_sha256(path)
    if record.get("bytes") != size or record.get("sha256") != digest:
        raise RuntimeError(f"{label} digest or byte count drifted")


def _validate_manifest(
    paths: ReplayPaths,
    manifest: dict[str, object],
    *,
    authority: dict[str, object],
    books: Sequence[object],
) -> None:
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION
        or manifest.get("artifact_id") != ARTIFACT_ID
        or manifest.get("authority") != authority
        or manifest.get("authority_sha256") != _authority_digest(authority)
        or manifest.get("scorer_journal") != _scorer_journal_contract()
        or manifest.get("input_boundary", {}).get("partition") != "public TRAIN"
        or manifest.get("input_boundary", {}).get("held_out_books_materialized") != 0
        or manifest.get("input_boundary", {}).get("c1_inputs_materialized") != 0
    ):
        raise RuntimeError("replay manifest authority or public-TRAIN boundary drifted")
    marker = _validate_spend_marker(paths, authority)
    marker_record = manifest.get("spend_marker")
    marker_sha256, _marker_bytes = _path_sha256(paths.marker)
    if (
        not isinstance(marker_record, dict)
        or marker_record.get("file") != paths.marker.name
        or marker_record.get("sha256") != marker_sha256
        or marker_record.get("state") != marker["state"]
    ):
        raise RuntimeError("replay manifest spend-marker binding drifted")
    cursor_record = manifest.get("cursor")
    rows = manifest.get("books")
    if not isinstance(cursor_record, dict) or not isinstance(rows, list):
        raise RuntimeError("replay cursor or book ledger is invalid")
    cursor = cursor_record.get("next_book_ordinal")
    if type(cursor) is not int or not 0 <= cursor <= len(books) or len(rows) != cursor:
        raise RuntimeError("replay cursor is outside the authenticated TRAIN prefix")
    token_count = 0
    for ordinal, (book, row) in enumerate(zip(books[:cursor], rows, strict=True)):
        if not isinstance(row, dict):
            raise RuntimeError("replay book row is invalid")
        if (
            row.get("ordinal") != ordinal
            or row.get("book") != book.short_name
            or row.get("tokens") != book.token_count
            or row.get("token_sha256_le_i32") != book.token_sha256_le_i32
        ):
            raise RuntimeError("replay book prefix drifted")
        shard_record = row.get("scorer_shard")
        if not isinstance(shard_record, dict):
            raise RuntimeError("scorer shard record is invalid")
        expected_shard = paths.scorer_shard(ordinal)
        shard_path = _manifest_sibling(
            paths.manifest, shard_record.get("file"), "scorer shard"
        )
        if shard_path != expected_shard:
            raise RuntimeError("scorer shard path drifted")
        if shard_record.get("records") != book.token_count:
            raise RuntimeError("scorer shard record count drifted")
        _validate_hashed_file(shard_path, shard_record, "scorer shard")
        token_count += int(book.token_count)
    if cursor_record.get("token_count") != token_count:
        raise RuntimeError("replay token cursor drifted")
    expected_status = _manifest_status(cursor, len(books))
    if manifest.get("status") != expected_status:
        raise RuntimeError("replay status and cursor disagree")
    checkpoint_record = manifest.get("checkpoint")
    if not isinstance(checkpoint_record, dict):
        raise RuntimeError("native checkpoint record is invalid")
    checkpoint_path = _manifest_sibling(
        paths.manifest, checkpoint_record.get("file"), "native checkpoint"
    )
    if checkpoint_path != paths.native_checkpoint(cursor):
        raise RuntimeError("native checkpoint cursor path drifted")
    _validate_hashed_file(checkpoint_path, checkpoint_record, "native checkpoint")
    identity = checkpoint_record.get("identity")
    summary = manifest.get("scorer_summary")
    if not isinstance(identity, dict) or not isinstance(summary, dict):
        raise RuntimeError("native/scorer state records are invalid")
    if (
        identity.get("segment_count") != summary.get("actual_allocated_segments")
        or manifest.get("actual_allocated_segments")
        != summary.get("actual_allocated_segments")
        or summary.get("observed_steps") != token_count
    ):
        raise RuntimeError("native/scorer/cursor accounting drifted")


def _remove_manifest_bound_next_generation_residue(
    paths: ReplayPaths,
    manifest: dict[str, object],
    *,
    authority: dict[str, object],
    next_book_ordinal: int,
) -> None:
    """Remove only the exact next generation under a validated run cursor."""

    if (
        manifest.get("authority") != authority
        or manifest.get("authority_sha256") != _authority_digest(authority)
        or manifest.get("cursor", {}).get("next_book_ordinal")
        != next_book_ordinal
    ):
        raise RuntimeError("cannot classify next-generation residue under this cursor")
    _validate_spend_marker(paths, authority)
    for path in (
        paths.native_checkpoint(next_book_ordinal + 1),
        paths.scorer_shard(next_book_ordinal),
    ):
        path.unlink(missing_ok=True)


def _token_columns(
    definition: NativeTemporalCircuitDefinition,
) -> dict[int, tuple[int, ...]]:
    columns = {int(token): tuple(values) for token, values in definition.token_codes}
    if len(columns) != len(definition.token_codes):
        raise RuntimeError("definition token ids are not unique")
    return columns


def _shard_header(
    *, book_ordinal: int, record_count: int, authority_sha256: str
) -> bytes:
    return SCORER_SHARD_HEADER.pack(
        SCORER_SHARD_MAGIC,
        SHARD_SCHEMA_VERSION,
        book_ordinal,
        record_count,
        bytes.fromhex(authority_sha256),
    )


def _pack_scorer_record(step: object, winners: tuple[int, ...]) -> bytes:
    values = (
        int(step.token_id),
        int(step.matching_segments),
        int(step.active_segments),
        int(step.punished_segments),
        int(step.allocated_segments),
        int(step.predicted_columns),
        *winners,
    )
    try:
        return SCORER_RECORD.pack(*values)
    except struct.error as error:
        raise RuntimeError("accepted scorer step exceeds compact journal bounds") from error


def _iter_scorer_shard(
    path: Path,
    *,
    expected_book_ordinal: int,
    expected_records: int,
    authority_sha256: str,
) -> Iterator[tuple[int, ...]]:
    expected_size = SCORER_SHARD_HEADER_BYTES + expected_records * SCORER_RECORD_BYTES
    if path.stat().st_size != expected_size:
        raise RuntimeError("scorer shard length is not canonical")
    with path.open("rb") as stream:
        raw_header = stream.read(SCORER_SHARD_HEADER_BYTES)
        if len(raw_header) != SCORER_SHARD_HEADER_BYTES:
            raise RuntimeError("scorer shard header is truncated")
        magic, version, ordinal, records, authority_bytes = SCORER_SHARD_HEADER.unpack(
            raw_header
        )
        if (
            magic != SCORER_SHARD_MAGIC
            or version != SHARD_SCHEMA_VERSION
            or ordinal != expected_book_ordinal
            or records != expected_records
            or authority_bytes.hex() != authority_sha256
        ):
            raise RuntimeError("scorer shard header authority drifted")
        for _index in range(expected_records):
            raw = stream.read(SCORER_RECORD_BYTES)
            if len(raw) != SCORER_RECORD_BYTES:
                raise RuntimeError("scorer shard record is truncated")
            yield SCORER_RECORD.unpack(raw)
        if stream.read(1):
            raise RuntimeError("scorer shard has trailing bytes")


def _restore_scorer(
    paths: ReplayPaths,
    manifest: dict[str, object],
    *,
    definition: NativeTemporalCircuitDefinition,
    books: Sequence[object],
    authority: dict[str, object],
) -> tuple[scoring.FactorizedSuccessorScorer, int]:
    started_ns = time.perf_counter_ns()
    scorer = scoring.FactorizedSuccessorScorer()
    codebook = _token_columns(definition)
    authority_sha256 = _authority_digest(authority)
    for ordinal, (book, row) in enumerate(
        zip(books[: manifest["cursor"]["next_book_ordinal"]], manifest["books"], strict=True)
    ):
        shard_path = paths.scorer_shard(ordinal)
        for token_index, values in enumerate(
            _iter_scorer_shard(
                shard_path,
                expected_book_ordinal=ordinal,
                expected_records=int(book.token_count),
                authority_sha256=authority_sha256,
            )
        ):
            token, matching, active, punished, allocated, predicted, *winners = values
            columns = codebook.get(token)
            if columns is None:
                raise RuntimeError("scorer journal token is outside the source-bound codebook")
            step = SimpleNamespace(
                token_id=token,
                event_count=token_index + 1,
                candidate_segments=matching,
                matching_segments=matching,
                active_segments=active,
                postings_visited=matching,
                allocated_segments=allocated,
                predicted_columns=predicted,
                punished_segments=punished,
                winner_spikes=tuple(
                    SimpleNamespace(cell_id=winner) for winner in winners
                ),
            )
            scorer.observe(
                step,
                columns=columns,
                reset_before=token_index == 0,
            )
        if row.get("scorer_anchor") != _scorer_anchor(scorer):
            raise RuntimeError("restored scorer state diverged from its book anchor")
    if scorer.summary() != manifest["scorer_summary"]:
        raise RuntimeError("restored scorer summary diverged from manifest")
    return scorer, time.perf_counter_ns() - started_ns


def _open_shard_temp(path: Path, book_ordinal: int, authority_sha256: str) -> tuple[BinaryIO, Path]:
    descriptor, raw_temp = tempfile.mkstemp(
        prefix=f"{path.name}.", suffix=".tmp", dir=path.parent
    )
    stream = os.fdopen(descriptor, "w+b")
    stream.write(
        _shard_header(
            book_ordinal=book_ordinal,
            record_count=0,
            authority_sha256=authority_sha256,
        )
    )
    return stream, Path(raw_temp)


def _run_book(
    circuit: NativeTemporalCircuit,
    scorer: scoring.FactorizedSuccessorScorer,
    book: object,
    tokens: np.ndarray,
    *,
    book_ordinal: int,
    global_start: int,
    shard_path: Path,
    columns_by_token: dict[int, tuple[int, ...]],
    authority_sha256: str,
    deadline_ns: int,
) -> tuple[dict[str, object], dict[str, object]]:
    if shard_path.exists():
        raise RuntimeError("next scorer shard still exists after cursor-bound cleanup")
    stream, temp_path = _open_shard_temp(shard_path, book_ordinal, authority_sha256)
    accepted = 0
    allocated_total = 0
    matching_total = 0
    active_total = 0
    started_ns = time.perf_counter_ns()
    try:
        for block_start in range(0, int(tokens.size), BLOCK_TOKENS):
            normalized = tuple(int(token) for token in tokens[block_start : block_start + BLOCK_TOKENS])
            steps = circuit.advance_block(
                normalized,
                mode=TemporalStepMode.LEARN,
                reset_before=block_start == 0,
            )
            if len(steps) != len(normalized):
                raise RuntimeError("native replay block returned an incomplete step tuple")
            for offset, (token, step) in enumerate(zip(normalized, steps, strict=True)):
                token_index = block_start + offset
                if int(step.token_id) != token:
                    raise RuntimeError("native replay step token drifted")
                columns = columns_by_token.get(token)
                if columns is None:
                    raise RuntimeError("public TRAIN token is outside the source-bound codebook")
                try:
                    winners = scorer.observe(
                        step,
                        columns=columns,
                        reset_before=token_index == 0,
                    )
                except scoring.FactorizedSuccessorStop as error:
                    raise ReplayAdjudicationStop(
                        {
                            "kind": "FIRST_FACTORIZED_SUCCESSOR_FAILURE",
                            "scorer_stop_kind": error.kind,
                            "scorer_step_index": error.step_index,
                            "details": error.details,
                            "book_ordinal": book_ordinal,
                            "book_short_name": str(book.short_name),
                            "book_zero_based_token_index": token_index,
                            "global_zero_based_step_index": global_start + token_index,
                            "token_id": token,
                            "native_block_overshoot_steps": len(steps) - offset - 1,
                            "accepted_scorer_summary_before_failure": scorer.summary(),
                        }
                    ) from error
                stream.write(_pack_scorer_record(step, winners))
                accepted += 1
                allocated_total += int(step.allocated_segments)
                matching_total += int(step.matching_segments)
                active_total += int(step.active_segments)
            if time.perf_counter_ns() > deadline_ns:
                raise RuntimeError("factorized successor chunk exceeded its wall cap")
        if accepted != int(tokens.size):
            raise RuntimeError("scorer shard did not accept the complete book")
        stream.seek(0)
        stream.write(
            _shard_header(
                book_ordinal=book_ordinal,
                record_count=accepted,
                authority_sha256=authority_sha256,
            )
        )
        stream.flush()
        os.fsync(stream.fileno())
        stream.close()
        os.replace(temp_path, shard_path)
    finally:
        if not stream.closed:
            stream.close()
        temp_path.unlink(missing_ok=True)
    digest, size = _path_sha256(shard_path)
    wall_ns = time.perf_counter_ns() - started_ns
    row = {
        "ordinal": book_ordinal,
        "book": str(book.short_name),
        "tokens": int(book.token_count),
        "token_sha256_le_i32": str(book.token_sha256_le_i32),
        "actual_allocated_segments": allocated_total,
        "matching_segments_total": matching_total,
        "active_segments_total": active_total,
        "matching_minus_active_segments_total": matching_total - active_total,
        "replay_wall_ns": wall_ns,
        "steps_per_second": int(tokens.size) * 1e9 / wall_ns,
        "scorer_anchor": _scorer_anchor(scorer),
        "scorer_shard": {
            "file": shard_path.name,
            "bytes": size,
            "sha256": digest,
            "records": accepted,
            "bytes_per_record": SCORER_RECORD_BYTES,
        },
    }
    return row, scorer.summary()


def _run_chunk(
    paths: ReplayPaths,
    definition: NativeTemporalCircuitDefinition,
    books: Sequence[object],
    token_arrays: Sequence[np.ndarray],
    *,
    authority: dict[str, object],
    binding: dict[str, object],
    max_books: int,
    deadline_ns: int,
) -> dict[str, object]:
    if not 1 <= max_books <= MAX_CHUNK_BOOKS:
        raise ValueError("max_books")
    manifest = _read_manifest(paths)
    _validate_manifest(paths, manifest, authority=authority, books=books)
    if manifest["status"] == "REPLAY_COMPLETE":
        return manifest
    scorer, restore_ns = _restore_scorer(
        paths,
        manifest,
        definition=definition,
        books=books,
        authority=authority,
    )
    start = int(manifest["cursor"]["next_book_ordinal"])
    end = min(start + max_books, len(books))
    old_checkpoint = _manifest_sibling(
        paths.manifest, manifest["checkpoint"]["file"], "native checkpoint"
    )
    columns_by_token = _token_columns(definition)
    authority_sha256 = _authority_digest(authority)
    rows = list(manifest["books"])
    timing_books = list(manifest["timing"]["books"])
    cumulative_restore = int(manifest["timing"]["cumulative_scorer_restore_wall_ns"])
    cumulative_replay = int(manifest["timing"]["cumulative_replay_wall_ns"])
    cumulative_io = int(manifest["timing"]["cumulative_checkpoint_io_wall_ns"])
    with _replay_authority(binding):
        import_started_ns = time.perf_counter_ns()
        circuit = NativeTemporalCircuit.from_checkpoint_path(definition, old_checkpoint)
        import_ns = time.perf_counter_ns() - import_started_ns
        try:
            if _native_identity_record(circuit.checkpoint_identity()) != manifest["checkpoint"]["identity"]:
                raise RuntimeError("resumed native checkpoint identity drifted")
            for ordinal in range(start, end):
                book = books[ordinal]
                tokens = token_arrays[ordinal]
                new_checkpoint = paths.native_checkpoint(ordinal + 1)
                new_shard = paths.scorer_shard(ordinal)
                _remove_manifest_bound_next_generation_residue(
                    paths,
                    manifest,
                    authority=authority,
                    next_book_ordinal=ordinal,
                )
                row, summary = _run_book(
                    circuit,
                    scorer,
                    book,
                    tokens,
                    book_ordinal=ordinal,
                    global_start=sum(int(item.token_count) for item in books[:ordinal]),
                    shard_path=new_shard,
                    columns_by_token=columns_by_token,
                    authority_sha256=authority_sha256,
                    deadline_ns=deadline_ns,
                )
                identity = circuit.checkpoint_identity()
                if identity.segment_count != summary["actual_allocated_segments"]:
                    raise RuntimeError("native and factorized scorer allocation totals diverged")
                export_started_ns = time.perf_counter_ns()
                publication = circuit.publish_checkpoint(new_checkpoint)
                export_ns = time.perf_counter_ns() - export_started_ns
                if not publication.semantically_resumable:
                    raise RuntimeError("book-boundary native checkpoint is not resumable")
                checkpoint_record = _checkpoint_record(
                    new_checkpoint, identity, publication
                )
                rows.append(row)
                timing_books.append(
                    {
                        "ordinal": ordinal,
                        "resume_import_wall_ns": import_ns if ordinal == start else 0,
                        "scorer_restore_wall_ns": restore_ns if ordinal == start else 0,
                        "replay_wall_ns": row["replay_wall_ns"],
                        "checkpoint_export_wall_ns": export_ns,
                    }
                )
                cumulative_restore += restore_ns if ordinal == start else 0
                cumulative_replay += int(row["replay_wall_ns"])
                cumulative_io += (import_ns if ordinal == start else 0) + export_ns
                cursor = ordinal + 1
                manifest = {
                    **manifest,
                    "status": _manifest_status(cursor, len(books)),
                    "cursor": {
                        "next_book_ordinal": cursor,
                        "token_count": sum(int(item.token_count) for item in books[:cursor]),
                    },
                    "books": rows,
                    "scorer_summary": summary,
                    "actual_allocated_segments": summary["actual_allocated_segments"],
                    "checkpoint": checkpoint_record,
                    "timing": {
                        "cumulative_replay_wall_ns": cumulative_replay,
                        "cumulative_scorer_restore_wall_ns": cumulative_restore,
                        "cumulative_checkpoint_io_wall_ns": cumulative_io,
                        "books": timing_books,
                    },
                }
                _atomic_write_json(paths.manifest, manifest)
                old_checkpoint.unlink(missing_ok=True)
                old_checkpoint = new_checkpoint
                import_ns = 0
                restore_ns = 0
                print(
                    f"factorized TRAIN49 {cursor:02d}/{len(books):02d} "
                    f"{book.short_name}: segments={identity.segment_count:,} "
                    f"wall={row['replay_wall_ns'] / 1e9:.2f}s",
                    flush=True,
                )
        finally:
            circuit.close()
    _validate_manifest(paths, manifest, authority=authority, books=books)
    return manifest


def _preflight(
    paths: ReplayPaths,
    *,
    authority: dict[str, object],
    books: Sequence[object],
) -> dict[str, object]:
    if paths.output.exists():
        raise RuntimeError("terminal factorized successor artifact already exists")
    free = shutil.disk_usage(paths.state_base.parent).free
    if free < MIN_FREE_DISK_BYTES:
        raise RuntimeError(
            f"factorized replay requires at least {MIN_FREE_DISK_BYTES} free bytes"
        )
    if not paths.manifest.exists():
        _require_no_unmanifested_state(paths)
    if paths.marker.exists():
        _validate_spend_marker(paths, authority)
    manifest = None
    if paths.manifest.exists():
        manifest = _read_manifest(paths)
        _validate_manifest(paths, manifest, authority=authority, books=books)
    if manifest is not None and not paths.marker.exists():
        raise RuntimeError("replay manifest exists without its spend marker")
    return {
        "status": "PREFLIGHT_READY_NOT_RUN",
        "artifact_id": ARTIFACT_ID,
        "execution_started": False,
        "partition": "public TRAIN",
        "held_out_books_materialized": 0,
        "c1_inputs_materialized": 0,
        "spend_state": "UNSPENT" if not paths.marker.exists() else "SPENT_RESUMABLE",
        "replay_status": "NOT_STARTED" if manifest is None else manifest["status"],
        "next_book_ordinal": 0 if manifest is None else manifest["cursor"]["next_book_ordinal"],
        "free_disk_bytes": free,
        "minimum_free_disk_bytes": MIN_FREE_DISK_BYTES,
        "storage": {
            "maximum_native_checkpoint_bytes": MAX_NATIVE_CHECKPOINT_BYTES,
            "persistent_native_checkpoint_copies": 1,
            "temporary_native_checkpoint_copies_during_publication": 1,
            "maximum_scorer_journal_bytes": MAX_SCORER_JOURNAL_BYTES,
            "expected_final_durable_bytes": EXPECTED_FINAL_DURABLE_BYTES,
            "expected_peak_durable_bytes": EXPECTED_PEAK_DURABLE_BYTES,
        },
        "time_estimate": {
            "reference_dense_native_train49_replay_ns": REFERENCE_DENSE_NATIVE_TRAIN49_REPLAY_NS,
            "expected_total_hours": EXPECTED_REPLAY_HOURS,
            "basis": "measured dense native TRAIN49 replay plus Python scorer/restore/I-O headroom",
        },
        "resume_cost_boundary": {
            "restore_time_complexity_per_invocation": "O(authenticated prefix token records)",
            "restores_entire_scorer_prefix": True,
            "maximum_prefix_records": cn2.FULL_TRAIN_TOKENS,
            "maximum_complete_keys": cn2.FULL_TRAIN_TOKENS - cn2.FULL_TRAIN_BOOKS,
            "maximum_owner_column_entries": cn2.WORST_CASE_SEGMENT_CAPACITY,
            "live_python_scorer_memory_estimate_gib": {
                "lower": 3.0,
                "upper": 8.0,
                "hard_cap": False,
                "status": "engineering estimate; unmeasured before adjudication",
            },
            "note": (
                "the compact durable journal does not bound live CPython dict/tuple "
                "memory; provision memory separately from the native checkpoint and "
                "bank an infrastructure stop rather than swapping through the gate"
            ),
        },
        "scorer_journal": _scorer_journal_contract(),
        "authority_sha256": _authority_digest(authority),
    }


def _terminal_payload(
    paths: ReplayPaths,
    manifest: dict[str, object],
    *,
    verified: object,
    binding: dict[str, object],
    result: dict[str, object],
    status: str,
) -> dict[str, object]:
    manifest_sha256, manifest_bytes = _path_sha256(paths.manifest)
    marker_sha256, marker_bytes = _path_sha256(paths.marker)
    return {
        "schema_version": 1,
        "artifact_id": ARTIFACT_ID,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": status,
        "claim_boundary": CLAIM_BOUNDARY,
        "criterion": GATE_CONTRACT,
        "input": {
            "partition": "public TRAIN",
            "candidate_id": verified.candidate_id,
            "train_input_authority_sha256": verified.train_input_authority_sha256,
            "train_stream_sha256": verified.train_stream_sha256,
            "train_books_materialized": cn2.FULL_TRAIN_BOOKS,
            "train_tokens": cn2.FULL_TRAIN_TOKENS,
            "held_out_books_materialized": 0,
            "c1_inputs_materialized": 0,
        },
        "start_state": manifest["start_state"],
        "cursor": manifest["cursor"],
        "result": result,
        "scorer_summary": manifest["scorer_summary"],
        "scorer_journal": _scorer_journal_contract(),
        "native_checkpoint": manifest["checkpoint"],
        "timing": manifest["timing"],
        "durability": {
            "manifest_file": paths.manifest.name,
            "manifest_sha256": manifest_sha256,
            "manifest_bytes": manifest_bytes,
            "spend_marker_file": paths.marker.name,
            "spend_marker_sha256": marker_sha256,
            "spend_marker_bytes": marker_bytes,
            "active_native_checkpoint_copies": 1,
            "durable_source_column_map": False,
        },
        "source_binding": binding,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "native": loaded_library_identity().to_record(),
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--state-base", type=Path, default=STATE_BASE_PATH)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="validate source/TRAIN/storage/resume state without spending or executing",
    )
    parser.add_argument(
        "--run-chunk-books",
        type=int,
        metavar="N",
        help=f"spend/resume one bounded chunk of 1..{MAX_CHUNK_BOOKS} TRAIN books",
    )
    parser.add_argument(
        "--confirm-public-train-factorized-spend",
        action="store_true",
        help="explicitly confirm the newly named public-TRAIN-only adjudication",
    )
    args = parser.parse_args()
    if args.preflight_only == (args.run_chunk_books is not None):
        parser.error("choose exactly one of --preflight-only or --run-chunk-books N")
    if args.run_chunk_books is not None and not 1 <= args.run_chunk_books <= MAX_CHUNK_BOOKS:
        parser.error(f"--run-chunk-books must be in 1..{MAX_CHUNK_BOOKS}")
    if args.run_chunk_books is not None and not args.confirm_public_train_factorized_spend:
        parser.error("--confirm-public-train-factorized-spend is required")
    return args


def main() -> int:
    args = _parse_args()
    paths = ReplayPaths(args.output.resolve(), args.state_base.resolve())
    paths.output.parent.mkdir(parents=True, exist_ok=True)
    paths.state_base.parent.mkdir(parents=True, exist_ok=True)
    verified, books, token_arrays = _load_train()
    _protocol, definition = _definition()
    binding = _source_binding()
    authority = _authority_record(verified, definition, binding)
    preflight = _preflight(paths, authority=authority, books=books)
    if args.preflight_only:
        print(json.dumps(preflight, indent=2, sort_keys=True), flush=True)
        return 0

    marker = _load_or_create_spend_marker(paths, authority)
    if not paths.manifest.exists():
        _initialize_empty_manifest(
            paths,
            definition,
            books,
            authority=authority,
            binding=binding,
            marker=marker,
        )
    try:
        manifest = _run_chunk(
            paths,
            definition,
            books,
            token_arrays,
            authority=authority,
            binding=binding,
            max_books=args.run_chunk_books,
            deadline_ns=time.perf_counter_ns() + RUN_WALL_CAP_NS,
        )
    except ReplayAdjudicationStop as stop:
        manifest = _read_manifest(paths)
        payload = _terminal_payload(
            paths,
            manifest,
            verified=verified,
            binding=binding,
            result=stop.record,
            status="BANKED_FIRST_FACTORIZED_SUCCESSOR_FAILURE",
        )
        _exclusive_write_json(paths.output, payload)
        print(f"BANKED_FIRST_FACTORIZED_SUCCESSOR_FAILURE: {paths.output}", flush=True)
        return 2

    if manifest["status"] != "REPLAY_COMPLETE":
        print(
            f"REPLAY_PARTIAL: books={manifest['cursor']['next_book_ordinal']}/"
            f"{len(books)} manifest={paths.manifest}",
            flush=True,
        )
        return 0
    scorer, _restore_ns = _restore_scorer(
        paths,
        manifest,
        definition=definition,
        books=books,
        authority=authority,
    )
    final_summary = scorer.finalize(
        terminal_segment_count=int(manifest["checkpoint"]["identity"]["segment_count"])
    )
    if final_summary != manifest["scorer_summary"]:
        raise RuntimeError("terminal restored scorer summary drifted")
    payload = _terminal_payload(
        paths,
        manifest,
        verified=verified,
        binding=binding,
        result={
            "kind": "PUBLIC_TRAIN_FACTORIZED_SUCCESSORS_COMPLETE",
            "completed_books": len(books),
            "completed_token_steps": final_summary["observed_steps"],
            "terminal_segment_count": manifest["checkpoint"]["identity"]["segment_count"],
        },
        status="PASS_PUBLIC_TRAIN_FACTORIZED_SUCCESSOR_REPLAY_V1",
    )
    _exclusive_write_json(paths.output, payload)
    print(f"PASS_PUBLIC_TRAIN_FACTORIZED_SUCCESSOR_REPLAY_V1: {paths.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
