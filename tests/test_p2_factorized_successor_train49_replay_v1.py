from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from experiments.r3_pilot import (
    run_p2_factorized_successor_train49_replay_v1 as replay,
)
from humanbrain.native_temporal_runtime import NativeTemporalCircuitDefinition


def _mini_definition() -> NativeTemporalCircuitDefinition:
    _protocol, production = replay.cn2._definition()
    return NativeTemporalCircuitDefinition(
        replace(production.config, segment_capacity=22),
        production.token_codes[:4],
        production.threshold_offsets_mv,
    )


def _mini_inputs(
    definition: NativeTemporalCircuitDefinition,
) -> tuple[tuple[object, ...], tuple[np.ndarray, ...]]:
    token_ids = [int(token) for token, _columns in definition.token_codes]
    books = tuple(
        SimpleNamespace(
            ordinal=ordinal,
            short_name=f"book-{ordinal}",
            token_count=2,
            token_sha256_le_i32=f"{ordinal + 1:064x}",
        )
        for ordinal in range(2)
    )
    arrays = tuple(
        np.asarray(token_ids[2 * ordinal : 2 * ordinal + 2], dtype="<i4")
        for ordinal in range(2)
    )
    return books, arrays


def _mini_binding() -> dict[str, object]:
    return {
        "protocol_identity_sha256": "11" * 32,
        "source_bundle_identity_sha256": "22" * 32,
    }


def _mini_authority(
    definition: NativeTemporalCircuitDefinition,
    binding: dict[str, object],
) -> dict[str, object]:
    return {
        "artifact_id": replay.ARTIFACT_ID,
        "partition": "public TRAIN",
        "train_stream_sha256": "33" * 32,
        "books": 2,
        "tokens": 4,
        "definition_identity_sha256": definition.circuit_identity_sha256(),
        "protocol_identity_sha256": binding["protocol_identity_sha256"],
        "source_bundle_identity_sha256": binding["source_bundle_identity_sha256"],
        "start_state": "EMPTY_NATIVE_AND_SCORER",
        "scorer_record_format": replay.SCORER_RECORD.format,
        "scorer_record_bytes": replay.SCORER_RECORD_BYTES,
    }


def _paths(tmp_path: Path, name: str) -> replay.ReplayPaths:
    return replay.ReplayPaths(
        output=tmp_path / f"{name}.json",
        state_base=tmp_path / name,
    )


def _initialize(
    paths: replay.ReplayPaths,
    definition: NativeTemporalCircuitDefinition,
    books: tuple[object, ...],
    authority: dict[str, object],
    binding: dict[str, object],
) -> dict[str, object]:
    marker = replay._create_spend_marker(paths, authority)
    return replay._initialize_empty_manifest(
        paths,
        definition,
        books,
        authority=authority,
        binding=binding,
        marker=marker,
    )


def _flip_last_byte(path: Path) -> bytes:
    original = path.read_bytes()
    path.write_bytes(original[:-1] + bytes([original[-1] ^ 0x01]))
    return original


def test_contract_is_source_bound_public_train_only_and_compact() -> None:
    assert replay.ARTIFACT_ID == "P2-FACTORIZED-SUCCESSOR-TRAIN49-REPLAY-v1"
    assert replay.SCORER_RECORD.format == "<4I2B11I"
    assert replay.SCORER_RECORD_BYTES == 62
    assert replay.SCORER_RECORD_FIELDS == (
        "token_id",
        "matching_segments",
        "active_segments",
        "punished_segments",
        "allocated_segments",
        "predicted_columns",
        "winner_cells[11]",
    )
    assert replay.MAX_SCORER_JOURNAL_BYTES == (
        replay.cn2.FULL_TRAIN_TOKENS * 62
        + replay.cn2.FULL_TRAIN_BOOKS * replay.SCORER_SHARD_HEADER_BYTES
    )
    assert replay.GATE_CONTRACT["claim_boundary"] == {
        "authorizes_c1": False,
        "capability_evidence": False,
        "completed_build_reused": False,
        "held_out_books_materialized": 0,
        "threshold_selection": False,
        "transfer_evidence": False,
    }
    contract = replay._scorer_journal_contract()
    assert contract["durable_python_source_column_map"] is False
    assert contract["derived_on_restore"]["columns"] == (
        "source-bound immutable definition token code"
    )
    binding = replay._source_binding()
    runner = Path(replay.__file__).resolve().relative_to(replay.REPO_ROOT).as_posix()
    scorer = Path(replay.scoring.__file__).resolve().relative_to(replay.REPO_ROOT).as_posix()
    assert binding["files"][runner] == replay.cn2._canonical_lf_sha256(
        Path(replay.__file__).resolve()
    )
    assert binding["files"][scorer] == replay.cn2._canonical_lf_sha256(
        Path(replay.scoring.__file__).resolve()
    )
    assert not any(
        "run_c1" in path.lower() or "held_out" in path.lower()
        for path in binding["files"]
    )


def test_preflight_does_not_spend_or_create_native_state(tmp_path: Path) -> None:
    paths = _paths(tmp_path, "preflight")
    authority = {"source_bundle_identity_sha256": "22" * 32}
    books = tuple()

    result = replay._preflight(paths, authority=authority, books=books)

    assert result["status"] == "PREFLIGHT_READY_NOT_RUN"
    assert result["execution_started"] is False
    assert result["spend_state"] == "UNSPENT"
    assert result["held_out_books_materialized"] == 0
    assert result["c1_inputs_materialized"] == 0
    assert result["storage"]["persistent_native_checkpoint_copies"] == 1
    assert result["storage"]["maximum_scorer_journal_bytes"] == (
        replay.MAX_SCORER_JOURNAL_BYTES
    )
    assert result["resume_cost_boundary"]["restore_time_complexity_per_invocation"] == (
        "O(authenticated prefix token records)"
    )
    assert result["resume_cost_boundary"]["restores_entire_scorer_prefix"] is True
    assert result["resume_cost_boundary"]["live_python_scorer_memory_estimate_gib"] == {
        "lower": 3.0,
        "upper": 8.0,
        "hard_cap": False,
        "status": "engineering estimate; unmeasured before adjudication",
    }
    assert not paths.marker.exists()
    assert not paths.manifest.exists()
    assert not paths.native_checkpoint(0).exists()


def test_fresh_spend_guard_is_exclusive_and_source_bound(tmp_path: Path) -> None:
    paths = _paths(tmp_path, "spend")
    authority = {"source_bundle_identity_sha256": "22" * 32}

    replay._create_spend_marker(paths, authority)

    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        replay._create_spend_marker(paths, authority)
    with pytest.raises(RuntimeError, match="authority or contract drifted"):
        replay._validate_spend_marker(
            paths, {"source_bundle_identity_sha256": "44" * 32}
        )


def test_fresh_spend_and_empty_initialization_reject_ambiguous_state_residue(
    tmp_path: Path,
) -> None:
    authority = {"source_bundle_identity_sha256": "22" * 32}
    fresh_paths = _paths(tmp_path, "fresh-residue")
    residue = fresh_paths.scorer_shard(0)
    residue.write_bytes(b"ambiguous scorer state")

    with pytest.raises(RuntimeError, match="unmanifested native/scorer state residue"):
        replay._create_spend_marker(fresh_paths, authority)
    assert residue.read_bytes() == b"ambiguous scorer state"
    assert not fresh_paths.marker.exists()

    definition = _mini_definition()
    books, _arrays = _mini_inputs(definition)
    binding = _mini_binding()
    authority = _mini_authority(definition, binding)
    initialize_paths = _paths(tmp_path, "initialize-residue")
    marker = replay._create_spend_marker(initialize_paths, authority)
    native_residue = initialize_paths.native_checkpoint(0)
    native_residue.write_bytes(b"ambiguous native state")

    with pytest.raises(RuntimeError, match="unmanifested native/scorer state residue"):
        replay._initialize_empty_manifest(
            initialize_paths,
            definition,
            books,
            authority=authority,
            binding=binding,
            marker=marker,
        )
    assert native_residue.read_bytes() == b"ambiguous native state"
    assert not initialize_paths.manifest.exists()


def test_resume_reconstructs_exact_scorer_and_native_state_like_uninterrupted(
    tmp_path: Path,
) -> None:
    definition = _mini_definition()
    books, arrays = _mini_inputs(definition)
    binding = _mini_binding()
    authority = _mini_authority(definition, binding)
    deadline = time.perf_counter_ns() + 60_000_000_000

    uninterrupted_paths = _paths(tmp_path, "uninterrupted")
    _initialize(uninterrupted_paths, definition, books, authority, binding)
    uninterrupted = replay._run_chunk(
        uninterrupted_paths,
        definition,
        books,
        arrays,
        authority=authority,
        binding=binding,
        max_books=2,
        deadline_ns=deadline,
    )

    resumed_paths = _paths(tmp_path, "resumed")
    empty = _initialize(resumed_paths, definition, books, authority, binding)
    first = replay._run_chunk(
        resumed_paths,
        definition,
        books,
        arrays,
        authority=authority,
        binding=binding,
        max_books=1,
        deadline_ns=deadline,
    )
    resumed = replay._run_chunk(
        resumed_paths,
        definition,
        books,
        arrays,
        authority=authority,
        binding=binding,
        max_books=1,
        deadline_ns=deadline,
    )

    assert empty["status"] == "REPLAY_EMPTY"
    assert empty["start_state"] == {
        "native": "FRESH_EMPTY",
        "scorer": "FRESH_EMPTY",
        "completed_dense_checkpoint_imported": False,
    }
    assert empty["scorer_summary"]["observed_steps"] == 0
    assert empty["checkpoint"]["identity"]["segment_count"] == 0
    assert first["status"] == "REPLAY_PARTIAL"
    assert first["cursor"] == {"next_book_ordinal": 1, "token_count": 2}
    assert resumed["status"] == "REPLAY_COMPLETE"
    assert resumed["cursor"] == {"next_book_ordinal": 2, "token_count": 4}
    assert resumed["scorer_summary"] == uninterrupted["scorer_summary"]
    assert resumed["scorer_summary"]["actual_allocated_segments"] == 22
    assert resumed["scorer_summary"]["matching_active_telemetry"] == (
        uninterrupted["scorer_summary"]["matching_active_telemetry"]
    )

    resumed_scorer, _restore_ns = replay._restore_scorer(
        resumed_paths,
        resumed,
        definition=definition,
        books=books,
        authority=authority,
    )
    uninterrupted_scorer, _restore_ns = replay._restore_scorer(
        uninterrupted_paths,
        uninterrupted,
        definition=definition,
        books=books,
        authority=authority,
    )
    assert resumed_scorer.summary() == uninterrupted_scorer.summary()
    assert resumed_scorer.finalize(terminal_segment_count=22) == (
        uninterrupted_scorer.finalize(terminal_segment_count=22)
    )

    resumed_checkpoint = resumed_paths.native_checkpoint(2)
    uninterrupted_checkpoint = uninterrupted_paths.native_checkpoint(2)
    assert resumed_checkpoint.read_bytes() == uninterrupted_checkpoint.read_bytes()
    assert resumed["checkpoint"]["identity"] == uninterrupted["checkpoint"]["identity"]
    assert resumed["checkpoint"]["sha256"] == uninterrupted["checkpoint"]["sha256"]
    for ordinal, book in enumerate(books):
        shard = resumed_paths.scorer_shard(ordinal)
        assert shard.stat().st_size == (
            replay.SCORER_SHARD_HEADER_BYTES
            + book.token_count * replay.SCORER_RECORD_BYTES
        )


def test_resume_rejects_source_shard_and_checkpoint_drift(tmp_path: Path) -> None:
    definition = _mini_definition()
    books, arrays = _mini_inputs(definition)
    binding = _mini_binding()
    authority = _mini_authority(definition, binding)
    paths = _paths(tmp_path, "corrupt")
    _initialize(paths, definition, books, authority, binding)
    manifest = replay._run_chunk(
        paths,
        definition,
        books,
        arrays,
        authority=authority,
        binding=binding,
        max_books=1,
        deadline_ns=time.perf_counter_ns() + 60_000_000_000,
    )

    with pytest.raises(RuntimeError, match="authority or public-TRAIN boundary drifted"):
        replay._validate_manifest(
            paths,
            manifest,
            authority={**authority, "source_bundle_identity_sha256": "44" * 32},
            books=books,
        )

    shard = paths.scorer_shard(0)
    original_shard = _flip_last_byte(shard)
    with pytest.raises(RuntimeError, match="scorer shard digest or byte count drifted"):
        replay._validate_manifest(
            paths, manifest, authority=authority, books=books
        )
    shard.write_bytes(original_shard)

    checkpoint = paths.native_checkpoint(1)
    original_checkpoint = _flip_last_byte(checkpoint)
    with pytest.raises(RuntimeError, match="native checkpoint digest or byte count drifted"):
        replay._validate_manifest(
            paths, manifest, authority=authority, books=books
        )
    checkpoint.write_bytes(original_checkpoint)
    replay._validate_manifest(paths, manifest, authority=authority, books=books)

    forged = json.loads(paths.manifest.read_bytes())
    forged["checkpoint"]["identity"]["segment_count"] += 1
    replay._atomic_write_json(paths.manifest, forged)
    with pytest.raises(RuntimeError, match="native/scorer/cursor accounting drifted"):
        replay._validate_manifest(paths, forged, authority=authority, books=books)
