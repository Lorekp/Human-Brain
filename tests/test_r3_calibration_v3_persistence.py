from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from experiments.r3_pilot import calibration_v3_persistence as persistence
from experiments.r3_pilot import calibration_v3_protocol as protocol

ARTIFACT_ROOT = "experiments/r3_pilot/calibration_v3_artifacts"
RESUME_PATH = (
    f"{ARTIFACT_ROOT}/r3-f0/deterministic-confirmation/"
    "theta-m-18_theta-a-20/resume/transition-0000529340.resume.json"
)
CHECKPOINT_PATH = RESUME_PATH.removesuffix(".resume.json") + ".r3ntcp1"
RESULT_PATH = (
    f"{ARTIFACT_ROOT}/r3-f0/deterministic-confirmation/"
    "theta-m-18_theta-a-20/construction-seed-0000000017.json"
)
DEVELOPMENT_RESULT_PATH = (
    f"{ARTIFACT_ROOT}/r3-f0/development/theta-m-18_theta-a-20/construction-seed-0000000017.json"
)
EXPECTED_SOURCE_BUNDLE_SHA256 = "3" * 64


def _canonical_test_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _claim_boundary() -> dict[str, object]:
    return {
        "capability_evidence": False,
        "held_out_data_accessed": False,
        "transfer_evidence": False,
    }


def _source_binding(*, runnable: bool = True) -> dict[str, object]:
    return {"blockers": [] if runnable else ["runner disabled"], "runnable": runnable}


def _resume_cursor() -> dict[str, object]:
    return {
        "after_transition_event": 529340,
        "book_name": "Joshua",
        "book_ordinal_zero_based": 5,
        "event_count": 529346,
        "future_book_resets": 43,
        "last_token_index_zero_based": 42123,
        "learning_step": 529346,
        "next_token_index_zero_based": 42124,
        "remaining_transition_events": 1588023,
        "reset_before_next_token": False,
    }


def _validated_protocol() -> dict[str, object]:
    return {
        "artifact_contract": {
            "root": ARTIFACT_ROOT,
            "path_inventory": {
                "direct_parent_directories": 3,
                "full_parent_directory_closure": 7,
            },
            "r3_f0_train": {
                "development": {
                    "theta-m-18_theta-a-20": DEVELOPMENT_RESULT_PATH,
                },
                "deterministic_confirmation": {
                    "theta-m-18_theta-a-20": RESULT_PATH,
                },
                "deterministic_confirmation_resumes": [
                    CHECKPOINT_PATH,
                    RESUME_PATH,
                ],
            },
        },
        "screens": {
            "r3_f0_train": {
                "construction_seed_scope": {
                    "development": [17],
                    "deterministic_confirmation": [17],
                },
                "determinism": {"resume_milestones": [_resume_cursor()]},
            }
        },
        "claim_boundary": _claim_boundary(),
        "result_contract": {"artifact_creation_requires_runnable_source_binding": True},
        "source_binding": _source_binding(),
        "threshold_selection": {"candidate_arms": [[18, 20]]},
    }


def _protocol_sha256() -> str:
    return hashlib.sha256(_canonical_test_bytes(_validated_protocol())).hexdigest()


def _open_store(
    *,
    repository_root: Path,
    validated_protocol: dict[str, object],
) -> persistence.CalibrationV3ArtifactStore:
    return persistence.CalibrationV3ArtifactStore.open(
        repository_root=repository_root,
        validated_protocol=validated_protocol,
        expected_source_bundle_identity_sha256=EXPECTED_SOURCE_BUNDLE_SHA256,
    )


def _refresh_parent_inventory(frozen: dict[str, object]) -> None:
    contract = frozen["artifact_contract"]
    assert type(contract) is dict
    root = contract["root"]
    assert type(root) is str
    paths = persistence._collect_artifact_paths(contract, artifact_root=root)
    direct, closure = persistence._artifact_parent_directories(
        paths,
        artifact_root=root,
    )
    contract["path_inventory"] = {
        "direct_parent_directories": len(direct),
        "full_parent_directory_closure": len(closure),
    }


def _prepare_store_namespace(
    store: persistence.CalibrationV3ArtifactStore,
) -> None:
    requests = store.durable_directory_requests()
    repository_root, first_relative_directory = requests[0]
    repository_root.joinpath(*first_relative_directory.split("/")).parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    for repository_root, relative_directory in requests:
        repository_root.joinpath(*relative_directory.split("/")).mkdir(exist_ok=True)


def _resume_payload() -> dict[str, object]:
    return {
        "arm_slug": "theta-m-18_theta-a-20",
        "artifact_path": RESUME_PATH,
        "checkpoint": {"body_sha256": "1" * 64},
        "checkpoint_publication": {"outcome": "DURABLE_PUBLISHED"},
        "circuit_contract": {"active_columns": 11},
        "claim_boundary": _claim_boundary(),
        "construction_seed": 17,
        "cursor": _resume_cursor(),
        "phase": "deterministic_confirmation",
        "previous_state": {"book": "café ☃"},
        "protocol_sha256": _protocol_sha256(),
        "reference_trajectory_id": "fresh-seed-17",
        "screen_id": "r3_f0_train",
        "source_binding": _source_binding(),
        "source_bundle_identity_sha256": EXPECTED_SOURCE_BUNDLE_SHA256,
        "stream_prefixes": {"winner_count": 529340},
        "thresholds": {"theta_active": 20, "theta_match": 18},
        "timing": {
            "prior_cumulative_ns": 0,
            "session_active_ns": 12,
            "cumulative_active_ns": 12,
        },
        "train_input": {"held_out_access": "forbidden"},
    }


def _result_payload() -> dict[str, object]:
    return {
        "artifact_path": RESULT_PATH,
        "claim_boundary": _claim_boundary(),
        "construction_seeds": [17],
        "phase": "deterministic_confirmation",
        "protocol_sha256": _protocol_sha256(),
        "result": {
            "arm_slug": "theta-m-18_theta-a-20",
            "comparisons": {},
            "kind": "r3_f0_train",
            "reference": {},
            "resumes": [],
            "structural_metrics": {},
            "train_input": {},
            "verdict": "PASS_R3_F0_TRAIN_STRUCTURE",
        },
        "screen_id": "r3_f0_train",
        "source_binding": _source_binding(),
        "source_bundle_identity_sha256": EXPECTED_SOURCE_BUNDLE_SHA256,
        "thresholds": {"theta_active": 20, "theta_match": 18},
    }


def _raw_resume(payload: dict[str, object] | None = None) -> bytes:
    exact_payload = _resume_payload() if payload is None else payload
    return _canonical_test_bytes(
        {
            "artifact_id": persistence.RESUME_ARTIFACT_ID,
            "payload": exact_payload,
            "payload_sha256": hashlib.sha256(_canonical_test_bytes(exact_payload)).hexdigest(),
            "schema_version": 1,
        }
    )


def _artifact_path(root: Path, relative_path: str) -> Path:
    return root.joinpath(*relative_path.split("/"))


def _artifact_path_with_parent(root: Path, relative_path: str) -> Path:
    path = _artifact_path(root, relative_path)
    path.parent.mkdir(parents=True)
    return path


def test_store_derives_membership_from_the_validated_schema_four_protocol(
    tmp_path: Path,
) -> None:
    frozen = protocol.build_protocol()
    protocol.validate_protocol(frozen, require_runnable=False)

    store = _open_store(
        repository_root=tmp_path,
        validated_protocol=frozen,
    )

    assert store.require_resume_manifest_destination(RESUME_PATH).relative_path == RESUME_PATH
    directory_requests = store.durable_directory_requests()
    assert len(directory_requests) == 54
    assert directory_requests[0] == (tmp_path.absolute(), ARTIFACT_ROOT)
    assert len(set(directory_requests)) == len(directory_requests)
    assert [path.count("/") for _root, path in directory_requests] == sorted(
        path.count("/") for _root, path in directory_requests
    )


def test_store_rejects_parent_inventory_that_does_not_match_frozen_paths(
    tmp_path: Path,
) -> None:
    frozen = _validated_protocol()
    contract = frozen["artifact_contract"]
    assert type(contract) is dict
    inventory = contract["path_inventory"]
    assert type(inventory) is dict
    inventory["full_parent_directory_closure"] = 8

    with pytest.raises(
        persistence.ArtifactAuthorityStop,
        match="parent-directory inventory",
    ):
        _open_store(repository_root=tmp_path, validated_protocol=frozen)


def test_official_store_requires_an_external_source_bundle_authority(tmp_path: Path) -> None:
    frozen = protocol.build_protocol()
    protocol.validate_protocol(frozen, require_runnable=False)

    with pytest.raises(persistence.ArtifactAuthorityStop):
        persistence.CalibrationV3ArtifactStore.open(
            repository_root=tmp_path,
            validated_protocol=frozen,
        )


@pytest.mark.parametrize(
    ("phase", "replacement"),
    [
        ("development", None),
        ("deterministic_confirmation", None),
        ("deterministic_confirmation", RESULT_PATH),
    ],
)
def test_store_rejects_missing_or_string_valued_r3_phase_contracts(
    tmp_path: Path,
    phase: str,
    replacement: object,
) -> None:
    frozen = _validated_protocol()
    contract = dict(frozen["artifact_contract"])
    r3_contract = dict(contract["r3_f0_train"])
    if replacement is None:
        del r3_contract[phase]
    else:
        r3_contract[phase] = replacement
    contract["r3_f0_train"] = r3_contract
    frozen["artifact_contract"] = contract

    with pytest.raises(persistence.ArtifactAuthorityStop):
        _open_store(repository_root=tmp_path, validated_protocol=frozen)


def test_envelope_binds_protocol_source_claim_and_external_bundle_authorities(
    tmp_path: Path,
) -> None:
    store = _open_store(
        repository_root=tmp_path,
        validated_protocol=_validated_protocol(),
    )
    variants: list[dict[str, object]] = []

    wrong_protocol = _resume_payload()
    wrong_protocol["protocol_sha256"] = "4" * 64
    variants.append(wrong_protocol)

    wrong_source = _resume_payload()
    wrong_source["source_binding"] = {"blockers": [], "runnable": False}
    variants.append(wrong_source)

    wrong_claim = _resume_payload()
    wrong_claim["claim_boundary"] = {
        "capability_evidence": False,
        "held_out_data_accessed": True,
        "transfer_evidence": False,
    }
    variants.append(wrong_claim)

    wrong_bundle = _resume_payload()
    wrong_bundle["source_bundle_identity_sha256"] = "4" * 64
    variants.append(wrong_bundle)

    for payload in variants:
        with pytest.raises(persistence.EnvelopeStop):
            store.encode_resume(payload)


def test_nonrunnable_protocol_blocks_official_artifact_encoding(tmp_path: Path) -> None:
    frozen = _validated_protocol()
    frozen["source_binding"] = _source_binding(runnable=False)
    payload = _result_payload()
    payload["protocol_sha256"] = hashlib.sha256(_canonical_test_bytes(frozen)).hexdigest()
    payload["source_binding"] = _source_binding(runnable=False)
    store = _open_store(repository_root=tmp_path, validated_protocol=frozen)

    with pytest.raises(persistence.ArtifactAuthorityStop):
        store.encode_result(payload)


def test_unfrozen_result_family_has_no_official_json_codec(tmp_path: Path) -> None:
    electrical_path = f"{ARTIFACT_ROOT}/electrical-v3/development/seed-0000000003.json"
    frozen = _validated_protocol()
    contract = dict(frozen["artifact_contract"])
    contract["electrical_v3"] = {"development": [electrical_path]}
    frozen["artifact_contract"] = contract
    _refresh_parent_inventory(frozen)
    store = _open_store(repository_root=tmp_path, validated_protocol=frozen)
    payload = _result_payload()
    payload.update(
        {
            "artifact_path": electrical_path,
            "construction_seeds": [3],
            "phase": "development",
            "protocol_sha256": hashlib.sha256(_canonical_test_bytes(frozen)).hexdigest(),
            "result": {},
            "screen_id": "electrical_v3",
        }
    )

    with pytest.raises(persistence.ArtifactAuthorityStop):
        store.encode_result(payload)
    with pytest.raises(persistence.ArtifactAuthorityStop):
        store.decode(electrical_path, b"{}")


def test_resume_envelope_round_trips_as_ascii_only_canonical_json(tmp_path: Path) -> None:
    store = _open_store(
        repository_root=tmp_path,
        validated_protocol=_validated_protocol(),
    )

    raw = store.encode_resume(_resume_payload())

    assert b"caf\\u00e9 \\u2603" in raw
    assert b"\n" not in raw
    assert raw.isascii()
    decoded = store.decode(RESUME_PATH, raw)
    assert decoded.artifact_id == persistence.RESUME_ARTIFACT_ID
    assert decoded.schema_version == 1
    assert decoded.payload == _resume_payload()


@pytest.mark.parametrize("forbidden", [1.25, float("nan"), float("inf"), -float("inf")])
def test_encoder_rejects_every_floating_point_value(tmp_path: Path, forbidden: float) -> None:
    store = _open_store(
        repository_root=tmp_path,
        validated_protocol=_validated_protocol(),
    )
    payload = _resume_payload()
    payload["previous_state"] = {"nested": {"value": forbidden}}

    with pytest.raises(persistence.EnvelopeStop):
        store.encode_resume(payload)


def test_result_envelope_has_its_own_exact_top_level_payload_schema(tmp_path: Path) -> None:
    store = _open_store(
        repository_root=tmp_path,
        validated_protocol=_validated_protocol(),
    )
    raw = store.encode_result(_result_payload())

    assert store.decode(RESULT_PATH, raw).payload == _result_payload()
    missing = _result_payload()
    del missing["result"]
    with pytest.raises(persistence.EnvelopeStop):
        store.encode_result(missing)
    unknown = _result_payload()
    unknown["unfrozen"] = 1
    with pytest.raises(persistence.EnvelopeStop):
        store.encode_result(unknown)
    wrong_destination = _result_payload()
    wrong_destination["artifact_path"] = RESUME_PATH
    with pytest.raises(persistence.ArtifactPathStop):
        store.encode_result(wrong_destination)


def test_r3_f0_result_object_has_the_exact_frozen_inner_keys(tmp_path: Path) -> None:
    store = _open_store(
        repository_root=tmp_path,
        validated_protocol=_validated_protocol(),
    )
    missing = _result_payload()
    missing_result = dict(missing["result"])
    del missing_result["reference"]
    missing["result"] = missing_result
    unknown = _result_payload()
    unknown_result = dict(unknown["result"])
    unknown_result["unfrozen"] = 1
    unknown["result"] = unknown_result

    with pytest.raises(persistence.EnvelopeStop):
        store.encode_result(missing)
    with pytest.raises(persistence.EnvelopeStop):
        store.encode_result(unknown)


def test_r3_f0_destination_not_payload_claim_selects_the_inner_schema(tmp_path: Path) -> None:
    store = _open_store(
        repository_root=tmp_path,
        validated_protocol=_validated_protocol(),
    )
    forged = _result_payload()
    forged["screen_id"] = "electrical_v3"
    forged["result"] = {}

    with pytest.raises(persistence.EnvelopeStop):
        store.encode_result(forged)


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("screen_id", "electrical_v3"),
        ("phase", "development"),
        ("construction_seeds", [3]),
        ("thresholds", {"theta_active": 18, "theta_match": 16}),
    ],
)
def test_result_destination_binds_frozen_r3_identity_fields(
    tmp_path: Path,
    field: str,
    invalid: object,
) -> None:
    store = _open_store(
        repository_root=tmp_path,
        validated_protocol=_validated_protocol(),
    )
    forged = _result_payload()
    forged[field] = invalid

    with pytest.raises(persistence.EnvelopeStop):
        store.encode_result(forged)


def test_result_destination_binds_frozen_r3_arm_slug(tmp_path: Path) -> None:
    store = _open_store(
        repository_root=tmp_path,
        validated_protocol=_validated_protocol(),
    )
    forged = _result_payload()
    result = dict(forged["result"])
    result["arm_slug"] = "theta-m-16_theta-a-18"
    forged["result"] = result

    with pytest.raises(persistence.EnvelopeStop):
        store.encode_result(forged)


def test_resume_destination_binds_the_exact_frozen_milestone(tmp_path: Path) -> None:
    store = _open_store(
        repository_root=tmp_path,
        validated_protocol=_validated_protocol(),
    )
    for field, invalid in (
        ("arm_slug", "theta-m-16_theta-a-18"),
        ("thresholds", {"theta_active": 18, "theta_match": 16}),
    ):
        forged = _resume_payload()
        forged[field] = invalid
        with pytest.raises(persistence.EnvelopeStop):
            store.encode_resume(forged)

    forged_cursor = _resume_payload()
    cursor = dict(forged_cursor["cursor"])
    cursor["event_count"] = 529345
    forged_cursor["cursor"] = cursor
    with pytest.raises(persistence.EnvelopeStop):
        store.encode_resume(forged_cursor)


def test_binary_checkpoint_destination_rejects_json_encode_decode_and_read(
    tmp_path: Path,
) -> None:
    store = _open_store(
        repository_root=tmp_path,
        validated_protocol=_validated_protocol(),
    )
    raw = store.encode_result(_result_payload())
    forged = _result_payload()
    forged["artifact_path"] = CHECKPOINT_PATH

    with pytest.raises(persistence.ArtifactPathStop):
        store.encode_result(forged)
    with pytest.raises(persistence.ArtifactPathStop):
        store.decode(CHECKPOINT_PATH, raw)

    checkpoint = _artifact_path_with_parent(tmp_path, CHECKPOINT_PATH)
    checkpoint.write_bytes(raw)
    with pytest.raises(persistence.ArtifactPathStop):
        store.read(CHECKPOINT_PATH)


def test_known_envelope_integer_fields_reject_json_booleans(tmp_path: Path) -> None:
    store = _open_store(
        repository_root=tmp_path,
        validated_protocol=_validated_protocol(),
    )
    resume_variants: list[dict[str, object]] = []
    construction_seed = _resume_payload()
    construction_seed["construction_seed"] = True
    resume_variants.append(construction_seed)
    threshold = _resume_payload()
    threshold["thresholds"] = {"theta_active": 20, "theta_match": True}
    resume_variants.append(threshold)
    cursor = _resume_payload()
    cursor_row = dict(cursor["cursor"])
    cursor_row["after_transition_event"] = True
    cursor["cursor"] = cursor_row
    resume_variants.append(cursor)
    timing = _resume_payload()
    timing["timing"] = {
        "prior_cumulative_ns": True,
        "session_active_ns": 12,
        "cumulative_active_ns": 12,
    }
    resume_variants.append(timing)

    for payload in resume_variants:
        with pytest.raises(persistence.EnvelopeStop):
            store.encode_resume(payload)

    result_seed = _result_payload()
    result_seed["construction_seeds"] = [True]
    result_threshold = _result_payload()
    result_threshold["thresholds"] = {"theta_active": True, "theta_match": 18}
    for payload in (result_seed, result_threshold):
        with pytest.raises(persistence.EnvelopeStop):
            store.encode_result(payload)


def test_resume_timing_fields_require_exact_addition(tmp_path: Path) -> None:
    store = _open_store(
        repository_root=tmp_path,
        validated_protocol=_validated_protocol(),
    )
    forged = _resume_payload()
    forged["timing"] = {
        "prior_cumulative_ns": 7,
        "session_active_ns": 5,
        "cumulative_active_ns": 11,
    }

    with pytest.raises(persistence.EnvelopeStop):
        store.encode_resume(forged)


def test_envelope_identity_digests_are_exact_lowercase_sha256(tmp_path: Path) -> None:
    store = _open_store(
        repository_root=tmp_path,
        validated_protocol=_validated_protocol(),
    )
    for field in ("protocol_sha256", "source_bundle_identity_sha256"):
        for invalid in ("a" * 63, "A" * 64, "g" * 64, 7):
            resume = _resume_payload()
            resume[field] = invalid
            with pytest.raises(persistence.EnvelopeStop):
                store.encode_resume(resume)
            result = _result_payload()
            result[field] = invalid
            with pytest.raises(persistence.EnvelopeStop):
                store.encode_result(result)


def test_resume_identity_fields_are_the_exact_frozen_r3_f0_authority(
    tmp_path: Path,
) -> None:
    store = _open_store(
        repository_root=tmp_path,
        validated_protocol=_validated_protocol(),
    )
    variants = []
    for field, invalid in (
        ("screen_id", "electrical_v3"),
        ("phase", "development"),
        ("construction_seed", 18),
        ("arm_slug", ""),
        ("reference_trajectory_id", 3),
    ):
        payload = _resume_payload()
        payload[field] = invalid
        variants.append(payload)

    for payload in variants:
        with pytest.raises(persistence.EnvelopeStop):
            store.encode_resume(payload)


@pytest.mark.parametrize(
    "field",
    [
        "checkpoint",
        "checkpoint_publication",
        "circuit_contract",
        "claim_boundary",
        "cursor",
        "previous_state",
        "source_binding",
        "stream_prefixes",
        "thresholds",
        "timing",
        "train_input",
    ],
)
def test_resume_named_object_seams_reject_scalar_substitution(tmp_path: Path, field: str) -> None:
    store = _open_store(
        repository_root=tmp_path,
        validated_protocol=_validated_protocol(),
    )
    payload = _resume_payload()
    payload[field] = "not-an-object"

    with pytest.raises(persistence.EnvelopeStop):
        store.encode_resume(payload)


def test_result_known_string_and_object_seams_are_not_scalar_substitutable(
    tmp_path: Path,
) -> None:
    store = _open_store(
        repository_root=tmp_path,
        validated_protocol=_validated_protocol(),
    )
    for field, invalid in (
        ("screen_id", ""),
        ("phase", 3),
        ("claim_boundary", False),
        ("source_binding", []),
        ("thresholds", 1),
    ):
        payload = _result_payload()
        payload[field] = invalid
        with pytest.raises(persistence.EnvelopeStop):
            store.encode_result(payload)

    for field, invalid in (
        ("arm_slug", ""),
        ("comparisons", 1),
        ("reference", []),
        ("resumes", {}),
        ("structural_metrics", False),
        ("train_input", "not-an-object"),
        ("verdict", 1),
    ):
        payload = _result_payload()
        result = dict(payload["result"])
        result[field] = invalid
        payload["result"] = result
        with pytest.raises(persistence.EnvelopeStop):
            store.encode_result(payload)


def test_decoder_rejects_duplicate_keys_at_outer_and_nested_object_levels(
    tmp_path: Path,
) -> None:
    store = _open_store(
        repository_root=tmp_path,
        validated_protocol=_validated_protocol(),
    )
    raw = _raw_resume()
    duplicate_outer = raw.replace(
        b"{",
        b'{"artifact_id":"r3-f0-resume-manifest-v1",',
        1,
    )
    duplicate_nested = raw.replace(
        b'"timing":{',
        b'"timing":{"cumulative_active_ns":12,',
        1,
    )

    with pytest.raises(persistence.EnvelopeStop):
        store.decode(RESUME_PATH, duplicate_outer)
    with pytest.raises(persistence.EnvelopeStop):
        store.decode(RESUME_PATH, duplicate_nested)


def test_decoder_rejects_unknown_or_missing_fields_at_each_implemented_schema_level(
    tmp_path: Path,
) -> None:
    store = _open_store(
        repository_root=tmp_path,
        validated_protocol=_validated_protocol(),
    )
    envelope = json.loads(_raw_resume())
    outer_missing = dict(envelope)
    del outer_missing["payload_sha256"]
    outer_unknown = dict(envelope)
    outer_unknown["unfrozen"] = 1
    payload_missing = _resume_payload()
    del payload_missing["cursor"]
    payload_unknown = _resume_payload()
    payload_unknown["unfrozen"] = 1

    for raw in (
        _canonical_test_bytes(outer_missing),
        _canonical_test_bytes(outer_unknown),
        _raw_resume(payload_missing),
        _raw_resume(payload_unknown),
    ):
        with pytest.raises(persistence.EnvelopeStop):
            store.decode(RESUME_PATH, raw)


def test_encoder_normalizes_mixed_non_string_unknown_object_keys(tmp_path: Path) -> None:
    store = _open_store(
        repository_root=tmp_path,
        validated_protocol=_validated_protocol(),
    )
    outer = _resume_payload()
    outer[1] = 0
    outer["extra"] = 0
    nested = _resume_payload()
    nested["thresholds"] = {
        "theta_active": 20,
        "theta_match": 18,
        1: 0,
        "extra": 0,
    }

    for payload in (outer, nested):
        with pytest.raises(persistence.EnvelopeStop):
            store.encode_resume(payload)


def test_decoder_rejects_float_constant_bool_as_int_and_wrong_payload_hash(
    tmp_path: Path,
) -> None:
    store = _open_store(
        repository_root=tmp_path,
        validated_protocol=_validated_protocol(),
    )
    float_raw = _raw_resume().replace(b'"construction_seed":17', b'"construction_seed":17.0')
    nan_raw = _raw_resume().replace(b'"construction_seed":17', b'"construction_seed":NaN')
    infinity_raw = _raw_resume().replace(b'"construction_seed":17', b'"construction_seed":Infinity')
    negative_infinity_raw = _raw_resume().replace(
        b'"construction_seed":17', b'"construction_seed":-Infinity'
    )
    bool_schema = json.loads(_raw_resume())
    bool_schema["schema_version"] = True
    wrong_hash = json.loads(_raw_resume())
    wrong_hash["payload_sha256"] = "0" * 64

    for raw in (
        float_raw,
        nan_raw,
        infinity_raw,
        negative_infinity_raw,
        _canonical_test_bytes(bool_schema),
        _canonical_test_bytes(wrong_hash),
    ):
        with pytest.raises(persistence.EnvelopeStop):
            store.decode(RESUME_PATH, raw)


def test_decoder_rejects_noncanonical_unicode_trailing_bytes_and_oversize(
    tmp_path: Path,
) -> None:
    store = _open_store(
        repository_root=tmp_path,
        validated_protocol=_validated_protocol(),
    )
    canonical = _raw_resume()
    raw_unicode = canonical.replace(b"caf\\u00e9", "café".encode())

    for raw in (
        raw_unicode,
        canonical + b"\n",
        b"{" + b" " * persistence.MAX_ENVELOPE_BYTES,
        b"\xff",
    ):
        with pytest.raises(persistence.EnvelopeStop):
            store.decode(RESUME_PATH, raw)


@pytest.mark.parametrize(
    "raw",
    [
        b"[" * 2_000 + b"0" + b"]" * 2_000,
        b'{"integer":' + b"9" * 5_000 + b"}",
    ],
)
def test_decoder_normalizes_bounded_parser_resource_failures(
    tmp_path: Path,
    raw: bytes,
) -> None:
    store = _open_store(
        repository_root=tmp_path,
        validated_protocol=_validated_protocol(),
    )

    with pytest.raises(persistence.EnvelopeStop):
        store.decode(RESUME_PATH, raw)


def test_encoder_normalizes_deep_json_recursion_failure(tmp_path: Path) -> None:
    store = _open_store(
        repository_root=tmp_path,
        validated_protocol=_validated_protocol(),
    )
    nested: object = {"leaf": 0}
    for _ in range(2_000):
        nested = {"child": nested}
    payload = _resume_payload()
    payload["previous_state"] = nested

    with pytest.raises(persistence.EnvelopeStop):
        store.encode_resume(payload)


def test_resume_pair_requires_prepared_frozen_namespace_and_rejects_escape(
    tmp_path: Path,
) -> None:
    store = _open_store(
        repository_root=tmp_path,
        validated_protocol=_validated_protocol(),
    )

    with pytest.raises(
        persistence.ArtifactPathStop,
        match="not prepared before active time",
    ):
        store.prepare_absent_resume_pair(RESUME_PATH)
    assert list(tmp_path.iterdir()) == []

    _prepare_store_namespace(store)
    destination = store.prepare_absent_resume_pair(RESUME_PATH).manifest_path

    assert destination == _artifact_path(tmp_path, RESUME_PATH)
    assert destination.parent.is_dir()
    assert not destination.exists()
    with pytest.raises(persistence.ArtifactPathStop):
        store.prepare_absent_resume_pair("../escape.resume.json")
    with pytest.raises(persistence.ArtifactPathStop):
        store.prepare_absent_resume_pair(f"{ARTIFACT_ROOT}/not-frozen.resume.json")


def test_namespace_rejects_non_directory_parent(tmp_path: Path) -> None:
    (tmp_path / "experiments").mkdir()
    (tmp_path / "experiments" / "r3_pilot").write_bytes(b"not a directory")
    store = _open_store(
        repository_root=tmp_path,
        validated_protocol=_validated_protocol(),
    )

    with pytest.raises(persistence.ArtifactPathStop):
        store.prepare_absent_resume_pair(RESUME_PATH)


def test_namespace_rejects_symlink_or_reparse_parent(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    link_parent = tmp_path / "experiments" / "r3_pilot"
    link_parent.parent.mkdir()
    try:
        link_parent.symlink_to(outside, target_is_directory=True)
    except OSError as exc:  # pragma: no cover - host privilege limitation.
        pytest.skip(f"directory symlinks are unavailable: {exc}")
    store = _open_store(
        repository_root=tmp_path,
        validated_protocol=_validated_protocol(),
    )

    with pytest.raises(persistence.ArtifactPathStop):
        store.prepare_absent_resume_pair(RESUME_PATH)


def test_namespace_rejects_windows_reparse_metadata_without_host_privilege(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _open_store(
        repository_root=tmp_path,
        validated_protocol=_validated_protocol(),
    )
    _artifact_path_with_parent(tmp_path, RESUME_PATH)
    reparse_parent = tmp_path / "experiments" / "r3_pilot"
    real_lstat = persistence.os.lstat

    def lstat_with_reparse(path: object) -> object:
        observed = real_lstat(path)
        if Path(path) == reparse_parent:
            return SimpleNamespace(
                st_mode=observed.st_mode,
                st_file_attributes=0x400,
            )
        return observed

    monkeypatch.setattr(persistence.os, "lstat", lstat_with_reparse)

    with pytest.raises(persistence.ArtifactPathStop):
        store.prepare_absent_resume_pair(RESUME_PATH)


def test_bounded_reader_accepts_one_plain_file_and_rejects_oversize_or_nonregular(
    tmp_path: Path,
) -> None:
    valid_root = tmp_path / "valid"
    valid_root.mkdir()
    valid_store = _open_store(
        repository_root=valid_root,
        validated_protocol=_validated_protocol(),
    )
    valid_path = _artifact_path_with_parent(valid_root, RESUME_PATH)
    valid_path.write_bytes(valid_store.encode_resume(_resume_payload()))
    assert valid_store.read(RESUME_PATH).payload == _resume_payload()

    large_root = tmp_path / "large"
    large_root.mkdir()
    large_store = _open_store(
        repository_root=large_root,
        validated_protocol=_validated_protocol(),
    )
    large_path = _artifact_path_with_parent(large_root, RESUME_PATH)
    large_path.write_bytes(b"x" * (persistence.MAX_ENVELOPE_BYTES + 1))
    with pytest.raises(persistence.ArtifactPathStop):
        large_store.read(RESUME_PATH)

    directory_root = tmp_path / "directory"
    directory_root.mkdir()
    directory_store = _open_store(
        repository_root=directory_root,
        validated_protocol=_validated_protocol(),
    )
    directory_path = _artifact_path_with_parent(directory_root, RESUME_PATH)
    directory_path.mkdir()
    with pytest.raises(persistence.ArtifactPathStop):
        directory_store.read(RESUME_PATH)


def test_bounded_read_clean_close_oserror_is_normalized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _open_store(
        repository_root=tmp_path,
        validated_protocol=_validated_protocol(),
    )
    destination = _artifact_path_with_parent(tmp_path, RESUME_PATH)
    destination.write_bytes(store.encode_resume(_resume_payload()))
    real_close = persistence.os.close

    def close_then_fail(descriptor: int) -> None:
        real_close(descriptor)
        raise OSError("synthetic descriptor close failure")

    monkeypatch.setattr(persistence.os, "close", close_then_fail)

    with pytest.raises(persistence.ArtifactPathStop, match="close"):
        store.read(RESUME_PATH)


def test_bounded_read_preserves_primary_artifact_path_stop_when_close_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _open_store(
        repository_root=tmp_path,
        validated_protocol=_validated_protocol(),
    )
    destination = _artifact_path_with_parent(tmp_path, RESUME_PATH)
    destination.write_bytes(store.encode_resume(_resume_payload()))
    real_close = persistence.os.close
    real_fstat = persistence.os.fstat
    fstat_calls = 0

    def drifting_fstat(descriptor: int) -> object:
        nonlocal fstat_calls
        observed = real_fstat(descriptor)
        fstat_calls += 1
        if fstat_calls == 2:
            return SimpleNamespace(
                st_mode=observed.st_mode,
                st_dev=observed.st_dev,
                st_ino=observed.st_ino,
                st_size=observed.st_size + 1,
                st_mtime_ns=observed.st_mtime_ns,
                st_ctime_ns=observed.st_ctime_ns,
                st_file_attributes=getattr(observed, "st_file_attributes", 0),
            )
        return observed

    def close_then_fail(descriptor: int) -> None:
        real_close(descriptor)
        raise OSError("synthetic descriptor close failure")

    monkeypatch.setattr(persistence.os, "fstat", drifting_fstat)
    monkeypatch.setattr(persistence.os, "close", close_then_fail)

    with pytest.raises(
        persistence.ArtifactPathStop,
        match="changed during bounded read",
    ) as stopped:
        store.read(RESUME_PATH)
    assert stopped.value.__notes__ == ["artifact descriptor close also failed: OSError"]


def test_bounded_read_normalizes_descriptor_operation_oserror(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _open_store(
        repository_root=tmp_path,
        validated_protocol=_validated_protocol(),
    )
    destination = _artifact_path_with_parent(tmp_path, RESUME_PATH)
    destination.write_bytes(store.encode_resume(_resume_payload()))

    def fail_read(descriptor: int, count: int) -> bytes:
        del descriptor, count
        raise OSError("synthetic descriptor read failure")

    monkeypatch.setattr(persistence.os, "read", fail_read)

    with pytest.raises(persistence.ArtifactPathStop, match="descriptor operation"):
        store.read(RESUME_PATH)


def test_bounded_read_primary_stop_survives_baseexception_from_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _open_store(
        repository_root=tmp_path,
        validated_protocol=_validated_protocol(),
    )
    destination = _artifact_path_with_parent(tmp_path, RESUME_PATH)
    destination.write_bytes(store.encode_resume(_resume_payload()))
    real_close = persistence.os.close
    real_fstat = persistence.os.fstat
    fstat_calls = 0

    def drifting_fstat(descriptor: int) -> object:
        nonlocal fstat_calls
        observed = real_fstat(descriptor)
        fstat_calls += 1
        if fstat_calls == 2:
            return SimpleNamespace(
                st_mode=observed.st_mode,
                st_dev=observed.st_dev,
                st_ino=observed.st_ino,
                st_size=observed.st_size + 1,
                st_mtime_ns=observed.st_mtime_ns,
                st_ctime_ns=observed.st_ctime_ns,
                st_file_attributes=getattr(observed, "st_file_attributes", 0),
            )
        return observed

    def close_then_interrupt(descriptor: int) -> None:
        real_close(descriptor)
        raise KeyboardInterrupt("synthetic descriptor close interrupt")

    monkeypatch.setattr(persistence.os, "fstat", drifting_fstat)
    monkeypatch.setattr(persistence.os, "close", close_then_interrupt)

    with pytest.raises(
        persistence.ArtifactPathStop,
        match="changed during bounded read",
    ) as stopped:
        store.read(RESUME_PATH)
    assert stopped.value.__notes__ == ["artifact descriptor close also failed: KeyboardInterrupt"]


def test_bounded_reader_rejects_symlink_or_reparse_final(tmp_path: Path) -> None:
    store = _open_store(
        repository_root=tmp_path,
        validated_protocol=_validated_protocol(),
    )
    destination = _artifact_path_with_parent(tmp_path, RESUME_PATH)
    outside = tmp_path / "outside.json"
    outside.write_bytes(store.encode_resume(_resume_payload()))
    try:
        destination.symlink_to(outside)
    except OSError as exc:  # pragma: no cover - host privilege limitation.
        pytest.skip(f"file symlinks are unavailable: {exc}")

    with pytest.raises(persistence.ArtifactPathStop):
        store.read(RESUME_PATH)


def test_bounded_reader_rejects_windows_reparse_final_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _open_store(
        repository_root=tmp_path,
        validated_protocol=_validated_protocol(),
    )
    destination = _artifact_path_with_parent(tmp_path, RESUME_PATH)
    destination.write_bytes(store.encode_resume(_resume_payload()))
    real_lstat = persistence.os.lstat

    def lstat_with_reparse(path: object) -> object:
        observed = real_lstat(path)
        if Path(path) == destination:
            return SimpleNamespace(
                st_mode=observed.st_mode,
                st_file_attributes=0x400,
            )
        return observed

    monkeypatch.setattr(persistence.os, "lstat", lstat_with_reparse)

    with pytest.raises(persistence.ArtifactPathStop):
        store.read(RESUME_PATH)


@pytest.mark.parametrize(
    ("checkpoint_present", "manifest_present", "expected"),
    [
        (False, False, "ABSENT"),
        (True, False, "CHECKPOINT_ONLY"),
        (False, True, "MANIFEST_ONLY"),
        (True, True, "COMMITTED"),
    ],
)
def test_resume_pair_state_matrix_is_fail_closed(
    tmp_path: Path,
    checkpoint_present: bool,
    manifest_present: bool,
    expected: str,
) -> None:
    store = _open_store(
        repository_root=tmp_path,
        validated_protocol=_validated_protocol(),
    )
    checkpoint = _artifact_path(tmp_path, CHECKPOINT_PATH)
    manifest = _artifact_path(tmp_path, RESUME_PATH)
    checkpoint.parent.mkdir(parents=True)
    if checkpoint_present:
        checkpoint.write_bytes(b"checkpoint")
    if manifest_present:
        manifest.write_bytes(store.encode_resume(_resume_payload()))

    if expected == "ABSENT":
        prepared = store.prepare_absent_resume_pair(RESUME_PATH)
        assert prepared.checkpoint_path == checkpoint
        assert prepared.manifest_path == manifest
    elif expected == "CHECKPOINT_ONLY":
        with pytest.raises(persistence.ArtifactPairStop) as caught:
            store.prepare_absent_resume_pair(RESUME_PATH)
        assert caught.value.state is persistence.ArtifactPairState.CHECKPOINT_ONLY
    elif expected == "MANIFEST_ONLY":
        with pytest.raises(persistence.ArtifactPairStop) as caught:
            store.open_committed_resume_pair(RESUME_PATH)
        assert caught.value.state is persistence.ArtifactPairState.MANIFEST_ONLY
    else:
        committed = store.open_committed_resume_pair(RESUME_PATH)
        assert committed.checkpoint_path == checkpoint


def test_resume_pair_rejects_noncanonical_manifest_before_committed(tmp_path: Path) -> None:
    store = _open_store(
        repository_root=tmp_path,
        validated_protocol=_validated_protocol(),
    )
    checkpoint = _artifact_path(tmp_path, CHECKPOINT_PATH)
    manifest = _artifact_path(tmp_path, RESUME_PATH)
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"native checkpoint validation is deferred")
    manifest.write_bytes(b"not a canonical resume manifest")

    with pytest.raises(persistence.EnvelopeStop):
        store.open_committed_resume_pair(RESUME_PATH)


def test_narrow_resume_write_seam_derives_pair_and_rejects_any_existing_final(
    tmp_path: Path,
) -> None:
    store = _open_store(
        repository_root=tmp_path,
        validated_protocol=_validated_protocol(),
    )
    _prepare_store_namespace(store)

    assert not hasattr(store, "prepare_parent")
    assert store.require_r3_result_destination(RESULT_PATH).kind is (
        persistence.ArtifactDestinationKind.R3_RESULT
    )
    prepared = store.prepare_absent_resume_pair(RESUME_PATH)

    assert prepared.checkpoint_path == _artifact_path(tmp_path, CHECKPOINT_PATH)
    assert prepared.manifest_path == _artifact_path(tmp_path, RESUME_PATH)
    prepared.checkpoint_path.write_bytes(b"checkpoint")
    with pytest.raises(persistence.ArtifactPairStop) as stopped:
        store.prepare_absent_resume_pair(RESUME_PATH)
    assert stopped.value.state is persistence.ArtifactPairState.CHECKPOINT_ONLY


def test_narrow_result_write_seam_requires_prepared_parent_and_absent_final(
    tmp_path: Path,
) -> None:
    store = _open_store(
        repository_root=tmp_path,
        validated_protocol=_validated_protocol(),
    )

    with pytest.raises(persistence.ArtifactPathStop, match="parent namespace"):
        store.prepare_absent_result(RESULT_PATH)

    _prepare_store_namespace(store)
    result_path = store.prepare_absent_result(RESULT_PATH)
    assert result_path == _artifact_path(tmp_path, RESULT_PATH)
    result_path.write_bytes(b"existing immutable result")

    with pytest.raises(persistence.ArtifactPathStop, match="already exists"):
        store.prepare_absent_result(RESULT_PATH)


def test_committed_resume_open_reads_manifest_before_requiring_checkpoint(
    tmp_path: Path,
) -> None:
    store = _open_store(
        repository_root=tmp_path,
        validated_protocol=_validated_protocol(),
    )
    _prepare_store_namespace(store)
    prepared = store.prepare_absent_resume_pair(RESUME_PATH)
    prepared.manifest_path.write_bytes(b"not canonical")

    with pytest.raises(persistence.EnvelopeStop):
        store.open_committed_resume_pair(RESUME_PATH)

    prepared.manifest_path.write_bytes(store.encode_resume(_resume_payload()))
    with pytest.raises(persistence.ArtifactPairStop) as stopped:
        store.open_committed_resume_pair(RESUME_PATH)
    assert stopped.value.state is persistence.ArtifactPairState.MANIFEST_ONLY

    prepared.checkpoint_path.write_bytes(b"checkpoint")
    committed = store.open_committed_resume_pair(RESUME_PATH)
    assert committed.checkpoint_path == prepared.checkpoint_path
    assert committed.manifest.payload == _resume_payload()
    assert committed.manifest_bytes == store.encode_resume(_resume_payload())


def test_checkpoint_final_pin_confirms_the_same_final_after_manifest_publication(
    tmp_path: Path,
) -> None:
    store = _open_store(
        repository_root=tmp_path,
        validated_protocol=_validated_protocol(),
    )
    _prepare_store_namespace(store)
    prepared = store.prepare_absent_resume_pair(RESUME_PATH)
    prepared.checkpoint_path.write_bytes(b"checkpoint-v1")

    pin = store.pin_checkpoint_final(RESUME_PATH)
    prepared.manifest_path.write_bytes(store.encode_resume(_resume_payload()))

    assert store.confirm_checkpoint_final(pin) == prepared.checkpoint_path


def test_checkpoint_final_pin_rejects_deletion_after_manifest_publication(
    tmp_path: Path,
) -> None:
    store = _open_store(
        repository_root=tmp_path,
        validated_protocol=_validated_protocol(),
    )
    _prepare_store_namespace(store)
    prepared = store.prepare_absent_resume_pair(RESUME_PATH)
    prepared.checkpoint_path.write_bytes(b"checkpoint-v1")
    pin = store.pin_checkpoint_final(RESUME_PATH)
    prepared.manifest_path.write_bytes(store.encode_resume(_resume_payload()))

    prepared.checkpoint_path.unlink()

    with pytest.raises(persistence.ArtifactPathStop, match="unavailable"):
        store.confirm_checkpoint_final(pin)


def test_checkpoint_final_pin_rejects_replacement_after_manifest_publication(
    tmp_path: Path,
) -> None:
    store = _open_store(
        repository_root=tmp_path,
        validated_protocol=_validated_protocol(),
    )
    _prepare_store_namespace(store)
    prepared = store.prepare_absent_resume_pair(RESUME_PATH)
    prepared.checkpoint_path.write_bytes(b"checkpoint-v1")
    pin = store.pin_checkpoint_final(RESUME_PATH)
    prepared.manifest_path.write_bytes(store.encode_resume(_resume_payload()))

    replacement = prepared.checkpoint_path.with_name("replacement.r3ntcp1")
    replacement.write_bytes(b"checkpoint-v2")
    replacement.replace(prepared.checkpoint_path)

    with pytest.raises(persistence.ArtifactPathStop, match="identity changed"):
        store.confirm_checkpoint_final(pin)


def test_manifest_final_pin_rejects_replacement_during_resume_import(
    tmp_path: Path,
) -> None:
    store = _open_store(
        repository_root=tmp_path,
        validated_protocol=_validated_protocol(),
    )
    _prepare_store_namespace(store)
    prepared = store.prepare_absent_resume_pair(RESUME_PATH)
    prepared.checkpoint_path.write_bytes(b"checkpoint-v1")
    prepared.manifest_path.write_bytes(store.encode_resume(_resume_payload()))
    pair = store.open_committed_resume_pair(RESUME_PATH)

    replacement = prepared.manifest_path.with_name("replacement.resume.json")
    replacement.write_bytes(prepared.manifest_path.read_bytes())
    replacement.replace(prepared.manifest_path)

    with pytest.raises(persistence.ArtifactPathStop, match="manifest final identity changed"):
        store.confirm_manifest_final(pair.manifest_pin)


def test_confirm_exact_never_adopts_different_existing_bytes(tmp_path: Path) -> None:
    store = _open_store(
        repository_root=tmp_path,
        validated_protocol=_validated_protocol(),
    )
    _prepare_store_namespace(store)
    prepared = store.prepare_absent_resume_pair(RESUME_PATH)
    expected = store.encode_resume(_resume_payload())
    prepared.manifest_path.write_bytes(expected)

    assert store.confirm_exact(RESUME_PATH, expected).payload == _resume_payload()
    with pytest.raises(persistence.EnvelopeStop, match="differ"):
        store.confirm_exact(RESUME_PATH, expected + b"\n")


def test_confirm_resume_pair_rejects_manifest_path_replacement_during_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _open_store(
        repository_root=tmp_path,
        validated_protocol=_validated_protocol(),
    )
    _prepare_store_namespace(store)
    prepared = store.prepare_absent_resume_pair(RESUME_PATH)
    prepared.checkpoint_path.write_bytes(b"checkpoint")
    expected = store.encode_resume(_resume_payload())
    prepared.manifest_path.write_bytes(expected)
    real_lstat = persistence.os.lstat
    manifest_lstats = 0

    def lstat_with_atomic_replacement(path: object) -> object:
        nonlocal manifest_lstats
        observed = real_lstat(path)
        if Path(path) != prepared.manifest_path:
            return observed
        manifest_lstats += 1
        if manifest_lstats == 1:
            return observed
        return SimpleNamespace(
            st_mode=observed.st_mode,
            st_dev=observed.st_dev,
            st_ino=observed.st_ino + 1,
            st_size=observed.st_size,
            st_mtime_ns=observed.st_mtime_ns,
            st_ctime_ns=observed.st_ctime_ns,
            st_file_attributes=getattr(observed, "st_file_attributes", 0),
        )

    monkeypatch.setattr(persistence.os, "lstat", lstat_with_atomic_replacement)

    with pytest.raises(persistence.ArtifactPathStop, match=r"changed|identity"):
        store.confirm_committed_resume_pair(RESUME_PATH, expected)


def test_cumulative_active_wall_arithmetic_enforces_exact_cap_and_integer_types() -> None:
    fresh = persistence.CumulativeActiveWallTiming.complete_fresh_session(40)
    assert fresh.contract_id == "r3-cumulative-active-wall-v2"
    assert fresh.cap_ns == 54_000_000_000_000
    assert fresh.prior_cumulative_ns == 0
    assert fresh.session_active_ns == 40
    assert fresh.cumulative_active_ns == 40
    assert fresh.state is persistence.TimingState.WITHIN_CAP

    resumed = fresh.plus_completed_session(2)
    assert resumed.prior_cumulative_ns == 40
    assert resumed.session_active_ns == 2
    assert resumed.cumulative_active_ns == 42

    at_cap = persistence.CumulativeActiveWallTiming.complete_fresh_session(54_000_000_000_000)
    assert at_cap.cumulative_active_ns == 54_000_000_000_000
    assert at_cap.state is persistence.TimingState.WITHIN_CAP
    assert at_cap.plus_completed_session(1).state is persistence.TimingState.OVER_CAP

    with pytest.raises((TypeError, ValueError)):
        persistence.CumulativeActiveWallTiming.complete_fresh_session(True)
    with pytest.raises((TypeError, ValueError)):
        fresh.plus_completed_session(False)
    with pytest.raises((TypeError, ValueError)):
        persistence.CumulativeActiveWallTiming.complete_resumed_session(-1, 2)
    with pytest.raises((TypeError, ValueError)):
        persistence.CumulativeActiveWallTiming(
            prior_cumulative_ns=40,
            session_active_ns=2,
            cumulative_active_ns=41,
        )


def test_unclean_timing_is_indeterminate_and_cannot_recover_by_arithmetic() -> None:
    timing = persistence.CumulativeActiveWallTiming.complete_fresh_session(123)

    interrupted = timing.with_unclean_interruption()

    assert interrupted.prior_cumulative_ns == 123
    assert interrupted.session_active_ns is None
    assert interrupted.cumulative_active_ns is None
    assert interrupted.state is persistence.TimingState.TIMING_INDETERMINATE
    assert interrupted.plus_completed_session(10).state is (
        persistence.TimingState.TIMING_INDETERMINATE
    )


def test_fresh_and_resumed_timing_remain_separate_with_a_conservative_max() -> None:
    fresh = persistence.CumulativeActiveWallTiming.complete_fresh_session(100)
    resumed = persistence.CumulativeActiveWallTiming.complete_resumed_session(80, 50)

    comparison = persistence.ConfirmationTimingComparison(
        fresh=fresh,
        resumed=resumed,
    )

    assert comparison.fresh.cumulative_active_ns == 100
    assert comparison.resumed.cumulative_active_ns == 130
    assert comparison.conservative_cumulative_active_ns == 130
    assert comparison.state is persistence.TimingState.WITHIN_CAP

    indeterminate = persistence.ConfirmationTimingComparison(
        fresh=fresh,
        resumed=resumed.with_unclean_interruption(),
    )
    assert indeterminate.conservative_cumulative_active_ns is None
    assert indeterminate.state is persistence.TimingState.TIMING_INDETERMINATE


def _spikes(cell_base: int, tick_base: int) -> tuple[persistence.TemporalSpikeFrame, ...]:
    return tuple(
        persistence.TemporalSpikeFrame(cell_id=cell_base + index, tick=tick_base + index)
        for index in range(11)
    )


def _step_frame() -> persistence.TemporalStepFrame:
    return persistence.TemporalStepFrame(
        token_id=0x01020304,
        punished_segments=0x05060708,
        event_count=0x0102030405060708,
        candidate_segments=4,
        matching_segments=3,
        active_segments=2,
        postings_visited=5,
        predicted=True,
        predicted_columns=9,
        allocated_segments=10,
        inhibition_onset_ticks=tuple(range(11)),
        column_simulated_ticks=tuple(range(100, 111)),
        winner_spikes=_spikes(1000, 10),
        basket_spikes=_spikes(2000, 30),
        proximal_spikes=_spikes(3000, 50),
        proximal_relay_spikes=_spikes(4000, 70),
        winner_relay_spikes=_spikes(5000, 90),
        winner_context=tuple(range(10000, 10022)),
        proximal_context=tuple(range(20000, 20022)),
        context=tuple(range(30000, 30022)),
    )


def test_winner_row_is_exactly_eleven_pyramidal_winners_and_56_bytes() -> None:
    frame = persistence.WinnerChainFrame(
        token_id=0x01020304,
        event_count=0x0102030405060708,
        winner_context=tuple(range(11)),
    )

    row = frame.canonical_row()

    assert len(row) == 56
    assert row.hex() == (
        "0403020108070605040302010000000001000000020000000300000004000000"
        "05000000060000000700000008000000090000000a000000"
    )
    with pytest.raises((TypeError, ValueError)):
        persistence.WinnerChainFrame(
            token_id=1,
            event_count=1,
            winner_context=tuple(range(22)),
        )


def test_chain_frames_and_restored_counts_reject_booleans_as_integers() -> None:
    with pytest.raises((TypeError, ValueError)):
        persistence.WinnerChainFrame(
            token_id=True,
            event_count=1,
            winner_context=tuple(range(11)),
        )
    with pytest.raises((TypeError, ValueError)):
        persistence.TemporalSpikeFrame(cell_id=1, tick=False)
    with pytest.raises((TypeError, ValueError)):
        persistence.ResumableHashChain.restore(
            kind=persistence.HashChainKind.WINNER,
            count=True,
            digest_hex="0" * 64,
        )


def test_step_row_matches_the_independent_689_byte_vector() -> None:
    row = _step_frame().canonical_row()

    assert len(row) == 689
    assert row[:95].hex() == (
        "0403020108070605080706050403020104000000000000000300000000000000"
        "0200000000000000050000000000000001090a0000010002000300040005000600"
        "0700080009000a006400650066006700680069006a006b006c006d006e00"
    )
    assert hashlib.sha256(row).hexdigest() == (
        "64830e077fd6e3c617ca0729d331ebece9a3b514c534bfd84ab0d732cda0945c"
    )


def test_resumable_hash_chains_match_fixed_vectors_and_split_execution() -> None:
    winner = persistence.WinnerChainFrame(
        token_id=0x01020304,
        event_count=0x0102030405060708,
        winner_context=tuple(range(11)),
    )
    winner_start = persistence.ResumableHashChain.start(persistence.HashChainKind.WINNER)
    assert winner_start.hexdigest == (
        "193a9ab8600fa0852643fc33f7c6d8e5c27c02613873a9db1ac44bd33104cec7"
    )
    winner_once = winner_start.append_winner(winner)
    assert winner_once.hexdigest == (
        "292ddc5195a0fbe1ae9755a0bb0c0f4e7ef77aad5f7d2cb984e70585e2d09864"
    )

    second = persistence.WinnerChainFrame(
        token_id=9,
        event_count=10,
        winner_context=tuple(range(20, 31)),
    )
    uninterrupted = winner_once.append_winner(second)
    restored = persistence.ResumableHashChain.restore(
        kind=persistence.HashChainKind.WINNER,
        count=winner_once.count,
        digest_hex=winner_once.hexdigest,
    )
    assert restored.append_winner(second) == uninterrupted

    step_start = persistence.ResumableHashChain.start(persistence.HashChainKind.STEP)
    assert step_start.hexdigest == (
        "d6f93b566fd5a9bcdb9c73a5e80d7e7f1355434537b6e5447fbfb7b9b726c7c8"
    )
    step_once = step_start.append_step(_step_frame())
    assert step_once.hexdigest == (
        "8ed19305d148bbbbc7d90eb8736598d4d590b8f290e59c399419e810299484db"
    )
    uninterrupted_steps = step_once.append_step(_step_frame())
    restored_steps = persistence.ResumableHashChain.restore(
        kind=persistence.HashChainKind.STEP,
        count=step_once.count,
        digest_hex=step_once.hexdigest,
    )
    assert restored_steps.append_step(_step_frame()) == uninterrupted_steps
