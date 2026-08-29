from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from experiments.r3_pilot import run_p2_cost_dense_train49_v1 as cost
from humanbrain.native_temporal_runtime import NativeTemporalCircuitDefinition


def _mini_definition() -> NativeTemporalCircuitDefinition:
    _protocol, production = cost.cn2._definition()
    config = replace(production.config, segment_capacity=22)
    return NativeTemporalCircuitDefinition(
        config,
        production.token_codes[:4],
        production.threshold_offsets_mv,
    )


def test_dense_train49_contract_is_cost_only_and_public_train() -> None:
    assert cost.ARTIFACT_ID == "P2-COST-DENSE-TRAIN49-v1"
    assert cost.DENSE_STEPS_AFTER_EPOCH0_PER_ARM == 8_927_839
    assert cost.PER_ARM_SCHEDULE_STEPS == 11_045_251
    assert cost.MAX_CHECKPOINT_BYTES == 1_490_673_136
    assert cost.EXPECTED_PEAK_CHECKPOINT_BYTES == 4_472_019_408
    assert cost.GATE_CONTRACT["claim_boundary"] == {
        "authorizes_c1": False,
        "capability_evidence": False,
        "held_out_data_accessed": False,
        "identity_rescored": False,
        "threshold_selection": False,
        "transfer_evidence": False,
    }
    interpretation = json.loads(cost.CN2_INTERPRETATION_PATH.read_bytes())
    assert interpretation["cost_interpretation"]["replacement_measurement"] == (cost.ARTIFACT_ID)


def test_direct_projection_uses_epoch0_dense_remainder_and_checkpoint_io() -> None:
    projection = cost._project_schedule(
        setup_ns=1,
        epoch0_replay_ns=2,
        dense_learn_wall_ns=cost.GENESIS_TOKENS * 10,
        dense_readonly_wall_ns=cost.GENESIS_TOKENS * 20,
        maximum_export_ns=3,
        maximum_import_ns=4,
    )
    expected_ns = (
        3 * 1
        + 3 * 2
        + 3 * cost.DENSE_STEPS_AFTER_EPOCH0_PER_ARM * 20
        + 3 * cost.CHECKPOINT_EXPORTS_PER_ARM * 3
        + 3 * cost.RESUME_IMPORTS_PER_ARM * 4
    )
    assert projection["dense_genesis_conservative_ns_per_token"] == 20
    assert projection["projected_total_hours"] == expected_ns / 3_600_000_000_000
    assert projection["authorizes_c1"] is False
    contention_status, contention = cost._timing_verdict(projection, (0.99, 0.72, 0.98))
    assert contention_status == "STOP_P2_COST_DENSE_TRAIN49_CONTENTION"
    assert contention["cap_verdict_authoritative"] is False
    isolated_status, isolated = cost._timing_verdict(projection, (0.99, 0.95, 0.98))
    assert isolated_status == "PASS_P2_COST_DENSE_TRAIN49_DEVELOPMENT"
    assert isolated["cap_verdict_authoritative"] is True


def test_preflight_is_no_run_and_binds_the_live_runner(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    checkpoint = tmp_path / "base.checkpoint"
    _interpretation, preflight = cost._preflight(output, checkpoint)
    assert preflight["status"] == "PREFLIGHT_READY_NOT_RUN"
    assert preflight["execution_started"] is False
    assert preflight["held_out_data_accessed"] is False
    assert preflight["minimum_replay_cpu_wall_ratio"] == 0.90
    assert preflight["outputs_absent"] is True
    assert preflight["expected_peak_checkpoint_bytes"] == (cost.EXPECTED_PEAK_CHECKPOINT_BYTES)
    binding = cost._authority_binding()
    runner = Path(cost.__file__).resolve()
    runner_key = runner.relative_to(cost.REPO_ROOT).as_posix()
    assert binding["files"][runner_key] == cost.cn2._canonical_lf_sha256(runner)
    assert preflight["source_bundle_identity_sha256"] == binding["source_bundle_identity_sha256"]
    assert not output.exists()
    assert not checkpoint.exists()


def test_two_resumable_chunks_are_byte_exact_to_uninterrupted_native_build(
    tmp_path: Path,
) -> None:
    definition = _mini_definition()
    books = tuple(
        SimpleNamespace(short_name=f"book-{ordinal}", token_count=2) for ordinal in range(2)
    )
    token_ids = [token for token, _columns in definition.token_codes]
    token_arrays = tuple(
        np.asarray(token_ids[2 * ordinal : 2 * ordinal + 2], dtype="<i4") for ordinal in range(2)
    )
    binding = {
        "protocol_identity_sha256": "11" * 32,
        "source_bundle_identity_sha256": "22" * 32,
    }
    authority = {
        "books": 2,
        "tokens": 4,
        "train_stream_sha256": "33" * 32,
        "construction_seed": 17,
        "segment_capacity": 22,
        **binding,
    }
    deadline_ns = time.perf_counter_ns() + 30_000_000_000

    uninterrupted_manifest_path = tmp_path / "uninterrupted.partial.json"
    uninterrupted = cost._build_train_chunk(
        definition,
        books,
        token_arrays,
        base_checkpoint=tmp_path / "uninterrupted.checkpoint",
        manifest_path=uninterrupted_manifest_path,
        authority=authority,
        binding=binding,
        max_books=2,
        deadline_ns=deadline_ns,
    )

    resumed_manifest_path = tmp_path / "resumed.partial.json"
    first = cost._build_train_chunk(
        definition,
        books,
        token_arrays,
        base_checkpoint=tmp_path / "resumed.checkpoint",
        manifest_path=resumed_manifest_path,
        authority=authority,
        binding=binding,
        max_books=1,
        deadline_ns=deadline_ns,
    )
    resumed = cost._build_train_chunk(
        definition,
        books,
        token_arrays,
        base_checkpoint=tmp_path / "resumed.checkpoint",
        manifest_path=resumed_manifest_path,
        authority=authority,
        binding=binding,
        max_books=1,
        deadline_ns=deadline_ns,
    )
    idempotent = cost._build_train_chunk(
        definition,
        books,
        token_arrays,
        base_checkpoint=tmp_path / "resumed.checkpoint",
        manifest_path=resumed_manifest_path,
        authority=authority,
        binding=binding,
        max_books=1,
        deadline_ns=deadline_ns,
    )

    assert first["status"] == "BUILD_PARTIAL"
    assert first["cursor"] == {
        "next_book_ordinal": 1,
        "token_count": 2,
    }
    assert resumed["status"] == "BUILD_COMPLETE"
    assert resumed["cursor"]["next_book_ordinal"] == 2
    assert resumed["actual_allocated_segments"] == 22
    assert resumed["actual_allocated_segments"] == uninterrupted["actual_allocated_segments"]
    assert [row["actual_allocated_segments"] for row in resumed["books"]] == [11, 11]
    assert idempotent == resumed

    uninterrupted_checkpoint = cost._manifest_checkpoint_path(
        uninterrupted_manifest_path, uninterrupted
    )
    resumed_checkpoint = cost._manifest_checkpoint_path(resumed_manifest_path, resumed)
    assert uninterrupted_checkpoint.read_bytes() == resumed_checkpoint.read_bytes()
    assert resumed["checkpoint"]["sha256"] == uninterrupted["checkpoint"]["sha256"]
    assert resumed["checkpoint"]["identity"] == uninterrupted["checkpoint"]["identity"]
    for digest_field in ("protocol_identity_sha256", "source_bundle_identity_sha256"):
        with pytest.raises(RuntimeError, match="authority drifted"):
            cost._build_train_chunk(
                definition,
                books,
                token_arrays,
                base_checkpoint=tmp_path / "resumed.checkpoint",
                manifest_path=resumed_manifest_path,
                authority={**authority, digest_field: "44" * 32},
                binding=binding,
                max_books=1,
                deadline_ns=deadline_ns,
            )
