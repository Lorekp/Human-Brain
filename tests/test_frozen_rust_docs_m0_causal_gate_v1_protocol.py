from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from inspect import signature
from pathlib import Path

import pytest
from experiments.r3_pilot import frozen_rust_docs_m0_causal_gate_v1_protocol as protocol


def _synthetic_ready_and_unknown() -> tuple[
    protocol.ReadyUnspentReceiptV1,
    dict[str, object],
]:
    root = "a" * 64
    unknown_payload = {
        "factor": f"tok:{protocol.FROZEN_UNKNOWN_NONCE}",
        "selector_view_sha256": root,
        "association_projection_sha256": root,
        "selector_occurrences": 0,
        "bound_factor_occurrences": 0,
    }
    unknown = {
        **unknown_payload,
        "sha256": protocol._digest(protocol._UNKNOWN_CUE_NAMESPACE, unknown_payload),
    }
    ready_payload = {
        name: root
        for name in protocol.ReadyUnspentReceiptV1.__dataclass_fields__
        if name not in {"schema", "status", "claim_boundary", "work", "sha256"}
    }
    ready_payload.update(
        {
            "schema": protocol.MANIFEST_SCHEMA,
            "status": protocol.READY_UNSPENT,
            "claim_boundary": protocol.CLAIM_BOUNDARY,
            "candidate_source_set_sha256": protocol.CANDIDATE_SOURCE_SET_SHA256,
            "control_roster_sha256": protocol.CONTROL_ROSTER_SHA256,
            "batch_payload_sha256": protocol.BATCH_PAYLOAD_SHA256,
            "oracle_payload_sha256": protocol.ORACLE_PAYLOAD_SHA256,
            "executor_harness_sha256": protocol.rust_selected_action_executor.HARNESS_SHA256,
            "executor_batch_protocol_sha256": (
                protocol.rust_selected_action_executor.BATCH_PROTOCOL_SHA256
            ),
            "unknown_cue_absence_sha256": unknown["sha256"],
            "channel_action_handle": "ra_synthetic_channel",
            "channel_structural_sha256": root,
            "mutex_action_handle": "ra_synthetic_mutex",
            "task_projection_sha256s": {
                name: sha256(f"synthetic-task:{name}".encode()).hexdigest()
                for name, _text in protocol.TASK_SPECS
            },
            "work": protocol.ZERO_EVALUATION_WORK,
        }
    )
    ready = protocol.ReadyUnspentReceiptV1(
        **ready_payload,
        sha256=protocol._digest(
            protocol._READY_NAMESPACE,
            {**ready_payload, "work": protocol.asdict(protocol.ZERO_EVALUATION_WORK)},
        ),
    )
    ready.authenticate()
    return ready, unknown


def test_frozen_batch_is_exact_rust_2021_and_does_not_open_oracle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[Path] = []
    original = Path.read_bytes

    def observe(path: Path) -> bytes:
        observed.append(path)
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", observe)

    batch = protocol.load_frozen_batch()

    assert batch.cases == protocol.FROZEN_CASES
    assert batch.payload_sha256 == protocol.BATCH_PAYLOAD_SHA256
    assert batch.file_sha256 == protocol.EXPECTED_BATCH_FILE_SHA256
    assert (
        protocol._digest(
            protocol._BATCH_NAMESPACE,
            protocol.FROZEN_CASES,
        )
        == protocol.BATCH_PAYLOAD_SHA256
    )
    assert (
        sha256(protocol._canonical_document(protocol._BATCH_DOCUMENT)).hexdigest()
        == protocol.EXPECTED_BATCH_FILE_SHA256
    )
    assert protocol.rust_selected_action_executor.RUST_EDITION == "2021"
    assert protocol.DEFAULT_BATCH_PATH in observed
    assert protocol.DEFAULT_ORACLE_PATH not in observed


def test_oracle_commitment_is_derived_without_opening_oracle_artifact() -> None:
    assert protocol._valid_sha256(protocol.EXPECTED_ORACLE_FILE_SHA256)
    assert protocol._valid_sha256(protocol.ORACLE_PAYLOAD_SHA256)
    assert protocol.ORACLE_PAYLOAD_SHA256 != protocol.BATCH_PAYLOAD_SHA256
    assert not hasattr(protocol, "FROZEN_EXPECTED_OUTPUTS")
    assert not hasattr(protocol, "_ORACLE_DOCUMENT")


def test_ready_unspent_work_rejects_every_nonzero_evaluation_counter() -> None:
    protocol.ZERO_EVALUATION_WORK.authenticate_ready_unspent()

    for field in protocol.FrozenWorkCountersV1.__dataclass_fields__:
        changed = replace(protocol.ZERO_EVALUATION_WORK, **{field: 1})
        with pytest.raises(
            protocol.FrozenRustDocsM0GateError,
            match="zero evaluation work",
        ):
            changed.authenticate_ready_unspent()

        wrong_type = replace(protocol.ZERO_EVALUATION_WORK, **{field: False})
        with pytest.raises(
            protocol.FrozenRustDocsM0GateError,
            match="work counters",
        ):
            wrong_type.authenticate_ready_unspent()


def test_missing_manifest_stops_before_any_live_builder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built = False

    def forbidden_build(_batch: object) -> object:
        nonlocal built
        built = True
        raise AssertionError("live builder must remain unopened")

    monkeypatch.setattr(protocol, "DEFAULT_MANIFEST_PATH", tmp_path / "missing.json")
    monkeypatch.setattr(protocol, "_build_live", forbidden_build)

    with pytest.raises(protocol.FrozenRustDocsM0GateError, match="invalid JSON artifact"):
        protocol._prepare_ready_unspent()

    assert built is False


def test_synthetic_ready_manifest_round_trips_without_live_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ready, unknown = _synthetic_ready_and_unknown()
    manifest_payload = {
        "schema": protocol.MANIFEST_SCHEMA,
        "status": protocol.READY_UNSPENT,
        "claim_boundary": protocol.CLAIM_BOUNDARY,
        "implementation_commit": "0" * 40,
        "implementation_files": [],
        "batch": {},
        "oracle_commitment": {},
        "frozen_inputs": {},
        "unknown_cue_absence": unknown,
        "readiness": ready.record(),
    }
    manifest = {
        **manifest_payload,
        "manifest_sha256": protocol._digest(
            protocol._MANIFEST_NAMESPACE,
            manifest_payload,
        ),
    }
    path = tmp_path / "ready.manifest.json"
    path.write_bytes(protocol._canonical_document(manifest))
    monkeypatch.setattr(protocol, "DEFAULT_MANIFEST_PATH", path)

    loaded, raw = protocol._load_manifest()
    reconstructed = protocol._ready_receipt_from_record(loaded["readiness"])

    assert raw == protocol._canonical_document(manifest)
    assert reconstructed == ready


@pytest.mark.parametrize(
    ("substitution", "error"),
    (
        ("batch_case_count", "batch commitment"),
        ("batch_zero_value", "batch commitment"),
        ("oracle_zero_read_count", "oracle commitment"),
        ("frozen_input_source_bytes", "frozen task/candidate/control inputs"),
    ),
)
def test_committed_static_manifest_rejects_json_numeric_type_substitutions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    substitution: str,
    error: str,
) -> None:
    root = "a" * 64
    commit = "0" * 40
    ready, unknown = _synthetic_ready_and_unknown()
    expected_files = ({"path": "synthetic.py", "sha256": root},)
    expected_frozen_inputs = {
        "task_records": [],
        "task_set_sha256": root,
        "candidate_sources": [{"source_sha256": root, "source_bytes": 7}],
        "candidate_source_set_sha256": protocol.CANDIDATE_SOURCE_SET_SHA256,
        "control_roster": list(protocol.CONTROL_ROSTER),
        "control_roster_sha256": protocol.CONTROL_ROSTER_SHA256,
    }
    manifest_path = tmp_path / "static.manifest.json"
    monkeypatch.setattr(protocol, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(protocol, "DEFAULT_MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(protocol, "DEFAULT_BATCH_PATH", tmp_path / "public.batch.json")
    monkeypatch.setattr(protocol, "DEFAULT_ORACLE_PATH", tmp_path / "sealed.oracle.json")
    monkeypatch.setattr(protocol, "_implementation_records", lambda _commit: expected_files)
    monkeypatch.setattr(protocol, "_project_tasks", lambda: object())
    monkeypatch.setattr(
        protocol,
        "_frozen_inputs_record",
        lambda _tasks: protocol._plain(expected_frozen_inputs),
    )
    monkeypatch.setattr(protocol, "_oracle_is_tracked", lambda: None)
    monkeypatch.setattr(protocol, "_git_head", lambda: commit)
    committed_raw = b""
    monkeypatch.setattr(protocol, "_git_blob", lambda _commit, _path: committed_raw)

    def write_and_load(payload: dict[str, object]) -> tuple[dict[str, object], bytes]:
        nonlocal committed_raw
        manifest = {
            **payload,
            "manifest_sha256": protocol._digest(protocol._MANIFEST_NAMESPACE, payload),
        }
        committed_raw = protocol._canonical_document(manifest)
        manifest_path.write_bytes(committed_raw)
        return protocol._load_manifest()

    batch = protocol.FrozenBatchV1(
        protocol.FROZEN_CASES,
        protocol.BATCH_PAYLOAD_SHA256,
        protocol.EXPECTED_BATCH_FILE_SHA256,
    ).record()
    oracle_commitment = {
        "path": protocol.DEFAULT_ORACLE_PATH.relative_to(protocol.REPO_ROOT).as_posix(),
        "file_sha256": protocol.EXPECTED_ORACLE_FILE_SHA256,
        "payload_sha256": protocol.ORACLE_PAYLOAD_SHA256,
        "batch_payload_sha256": protocol.BATCH_PAYLOAD_SHA256,
        "runtime_read_count_at_freeze": 0,
    }
    baseline = {
        "schema": protocol.MANIFEST_SCHEMA,
        "status": protocol.READY_UNSPENT,
        "claim_boundary": protocol.CLAIM_BOUNDARY,
        "implementation_commit": commit,
        "implementation_files": protocol._plain(expected_files),
        "batch": batch,
        "oracle_commitment": oracle_commitment,
        "frozen_inputs": protocol._plain(expected_frozen_inputs),
        "unknown_cue_absence": unknown,
        "readiness": ready.record(),
    }
    loaded, raw = write_and_load(protocol._plain(baseline))
    protocol._authenticate_manifest_static(loaded, raw)

    forged = protocol._plain(baseline)
    if substitution == "batch_case_count":
        forged["batch"]["case_count"] = 4.0
    elif substitution == "batch_zero_value":
        forged["batch"]["cases"][0][-1] = False
    elif substitution == "oracle_zero_read_count":
        forged["oracle_commitment"]["runtime_read_count_at_freeze"] = False
    else:
        forged["frozen_inputs"]["candidate_sources"][0]["source_bytes"] = 7.0
    loaded, raw = write_and_load(forged)

    with pytest.raises(protocol.FrozenRustDocsM0GateError, match=error):
        protocol._authenticate_manifest_static(loaded, raw)


@pytest.mark.parametrize("wrong_zero", (False, 0.0))
def test_unknown_cue_zero_counts_require_exact_int_type(wrong_zero: object) -> None:
    _ready, unknown = _synthetic_ready_and_unknown()
    forged = protocol._plain(unknown)
    forged["selector_occurrences"] = wrong_zero
    forged["sha256"] = protocol._digest(
        protocol._UNKNOWN_CUE_NAMESPACE,
        {key: value for key, value in forged.items() if key != "sha256"},
    )

    with pytest.raises(protocol.FrozenRustDocsM0GateError, match="absence proof"):
        protocol._authenticate_unknown_cue_absence_record(forged)


def test_oracle_loader_rejects_unauthed_inputs_before_first_byte_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reads = 0
    original = Path.read_bytes

    def observe(path: Path) -> bytes:
        nonlocal reads
        if path == protocol.DEFAULT_ORACLE_PATH:
            reads += 1
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", observe)

    with pytest.raises(TypeError, match="exact prepared"):
        protocol._load_oracle_after_authenticated_execution(
            object(),  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
        )

    assert reads == 0


def test_frozen_claim_is_lexical_dispatch_not_coding_transfer() -> None:
    assert len(protocol.FROZEN_CANDIDATE_SOURCES) == 2
    assert protocol.CONTROL_ROSTER == (
        "primary",
        "contrast",
        "source_order_reversal",
        "cell_off",
        "plasticity_off",
        "bound_memory_splice",
        "shared_factor_thread",
        "unknown_factor",
        "single_factor_channel",
        "single_factor_mutex",
    )
    assert "lexical" in protocol.CLAIM_BOUNDARY
    assert "not source synthesis" in protocol.CLAIM_BOUNDARY
    assert "transfer" in protocol.CLAIM_BOUNDARY


def test_executor_authority_seam_accepts_registry_then_primary_recall_only() -> None:
    prepare = signature(
        protocol.rust_selected_action_executor.prepare_rust_selected_action_executor_v1
    )
    execute = signature(
        protocol.rust_selected_action_executor.PreparedRustSelectedActionExecutorV1.execute_selected_once
    )

    assert tuple(prepare.parameters) == ("registry", "batch")
    assert tuple(execute.parameters) == ("self", "recall")
    assert not hasattr(
        protocol.rust_selected_action_executor.PreparedRustSelectedActionExecutorV1,
        "execute_once",
    )


def test_supported_protocol_api_exposes_no_live_pre_spend_authority() -> None:
    assert not hasattr(protocol, "PreparedRustDocsM0GateV1")
    assert not hasattr(protocol, "prepare_ready_unspent")
    assert "_PreparedRustDocsM0GateV1" not in protocol.__all__
    assert "_prepare_ready_unspent" not in protocol.__all__
    assert "_LivePreparedInputsV1" not in protocol.__all__
    assert "_live" in protocol._PreparedRustDocsM0GateV1.__dataclass_fields__
    assert "live" not in protocol._PreparedRustDocsM0GateV1.__dataclass_fields__
